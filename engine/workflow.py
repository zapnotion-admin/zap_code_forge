"""
engine/workflow.py
Agent pipeline: SCAN → REASON → DECOMPOSE → [STEP LOOP] → SIMULATE → REVIEW → [RETRY LOOP]

Stage model assignments:
  Scan      — coder model    Fast file analysis + cross-file consistency
  Reason    — reasoner model Structured step plan with constraints + failure modes
  Execute   — coder model    One step at a time, re-reads files between steps
  Simulate  — coder model    Mentally steps through N ticks of execution (NEW)
  Review    — coder model    Final review of all written files
  Retry     — coder model    Multi-pass convergence loop (up to MAX_RETRY_PASSES)

v4 upgrades:
  - SIMULATE stage: mentally traces 5-8 ticks of execution before review,
    catching state-mutation and timing bugs that static review misses.
  - Constraint-rich REASON prompt: forces planner to emit CONSTRAINTS,
    EDGE_CASES, and FAILURE_MODES sections — not just task decomposition.
  - Multi-pass RETRY loop: retries up to MAX_RETRY_PASSES times, stopping
    early on PASS. Much higher convergence rate than single-shot retry.
  - Failure pattern injection: known bug patterns (per project type) are
    injected into planner and reviewer prompts as explicit checks.
"""

import re
import os
from engine.ollama_client import single_response, ensure_model, unload_model, resolve_model
from engine.logger import log
from engine.brief import read_brief, format_brief_for_prompt, append_run_summary
from engine.project_map import build_project_map_section, update_summaries
from engine.plan_parser import parse_steps, extract_plan_summary
from engine.step_state import StepState
from engine.step_executor import run_steps
from engine.failure_patterns import get_patterns_for_task
from engine.simulate import run_simulation, format_simulation_for_retry
from core.config import (
    MODEL_CODER, MODEL_REASONER, MODEL_FALLBACK,
    MAX_CTX_CODER, MAX_CTX_REASONER,
    TEMPERATURE_CODING, TEMPERATURE_PLANNING,
    PYTHON_IMPORT_CHECK,
)

# How many review->fix passes to attempt before giving up.
# 1 = original behaviour. 3 = much higher convergence rate.
MAX_RETRY_PASSES = 3


def extract_verdict(text: str) -> str:
    m = re.search(
        r"VERDICT[\s:]+\*{0,2}(PASS|NEEDS_CHANGES|FAIL)\*{0,2}",
        text, re.IGNORECASE,
    )
    if m:
        return m.group(1).upper()
    upper = text.upper()
    if "NEEDS_CHANGES" in upper or "NEEDS CHANGES" in upper:
        return "NEEDS_CHANGES"
    if "FAIL" in upper:
        return "FAIL"
    if "PASS" in upper:
        return "PASS"
    return "NEEDS_CHANGES"


def _swap_to(model: str, emit, cancel_check) -> bool:
    if cancel_check():
        return False
    log(f"[pipeline] VRAM swap -> {model}")
    unload_model()
    ensure_model(model)
    return not cancel_check()


def run_pipeline(
    task:              str,
    file_context:      str,
    project_dir:       str = "",
    context_files:     list = None,
    coder_model:       str = None,
    reasoner_model:    str = None,
    stable_mode:       bool = True,
    progress_callback  = None,
    cancel_check:      object = None,
    _history_block:    str = "",    # injected by outer_loop on re-runs
    _refinement_mode:  bool = False, # True on pass 2+: skip plan, refine existing files
    _frozen_spec:      dict = None,  # frozen spec from pass 1, used in refinement mode
) -> dict:

    def emit(stage: str, text: str) -> None:
        log(f"[pipeline] stage={stage} len={len(text)}")
        if progress_callback:
            progress_callback(stage, text)

    def is_cancelled() -> bool:
        return cancel_check is not None and cancel_check()

    coder    = resolve_model(coder_model    or MODEL_CODER,    MODEL_FALLBACK)
    reasoner = resolve_model(reasoner_model or MODEL_REASONER, MODEL_FALLBACK)
    log(f"[pipeline] coder={coder}  reasoner={reasoner}  stable={stable_mode}")

    mode_label = "Stable" if stable_mode else "Fast"
    emit("status", f"[{mode_label}] Coder: {coder} | Reasoner: {reasoner}")

    files_section = "FILES:\n" + file_context if file_context else "No files provided."
    brief_content    = read_brief(project_dir) if project_dir else ""
    brief_section    = format_brief_for_prompt(brief_content)

    project_map_section = build_project_map_section(
        project_dir, exclude_files=context_files or []
    ) if project_dir else ""
    if project_map_section:
        brief_section = brief_section + "\n" + project_map_section + "\n"

    _explicit = re.search(
        r"(?:save (?:it )?as|save to|call it|name it|file(?:name)? (?:is )?)\s+([\w\-]+\.\w+)",
        task, re.IGNORECASE
    ) or re.search(
        r"\b([\w\-]+\.(?:py|js|ts|html|css|sh|json|rs|cpp|c|java|go|rb|php))\b", task
    )
    target_file = _explicit.group(1).strip() if _explicit else None
    target_instruction = (
        f"\nIMPORTANT: The output file MUST be named '{target_file}'. "
        f"Use 'FILE: {target_file}' in your output. Do NOT use any other filename."
    ) if target_file else ""

    # -- Failure patterns: computed once, injected into planner + reviewer
    failure_patterns = get_patterns_for_task(task, context_files or [])
    if failure_patterns:
        log(f"[pipeline] Failure patterns loaded")

    # ------------------------------------------------------------------
    # REFINEMENT MODE (pass 2+)
    # Skip scan → plan → decompose → execute.
    # Read existing files from disk, run reviewer + retry loop against
    # the frozen spec from pass 1. This is pure honing — no rewriting.
    # ------------------------------------------------------------------
    if _refinement_mode and _frozen_spec:
        log(f"[pipeline] Refinement mode — skipping plan/execute, reviewing existing files")
        emit("status", "Refinement pass — reviewing existing files against spec...")
        return _run_refinement_pass(
            task            = task,
            frozen_spec     = _frozen_spec,
            project_dir     = project_dir,
            context_files   = context_files or [],
            coder           = coder,
            emit            = emit,
            is_cancelled    = is_cancelled,
            failure_patterns= failure_patterns,
            brief_section   = brief_section,
        )

    # ------------------------------------------------------------------
    # STAGE 1: SCAN
    # ------------------------------------------------------------------
    has_files = bool(
        file_context
        and file_context.strip()
        and file_context.strip() != "No files provided."
    )

    if has_files:
        emit("status", "Scanning files...")
        ensure_model(coder)
        if is_cancelled():
            return {}

        ctx_filenames = [os.path.basename(f) for f in (context_files or [])]
        ctx_list = ", ".join(ctx_filenames) if ctx_filenames else "(none listed)"

        pattern_scan_hint = (
            f"\nAlso check for these known failure patterns:\n{failure_patterns}"
            if failure_patterns else ""
        )

        scan_prompt = f"""
TASK: {task}{target_instruction}
{brief_section}{files_section}

You are a code scanner. Find REAL problems in the existing files above.

Check:
1. Logic bugs visible in the code (wrong variable names, off-by-one errors, etc.)
2. Cross-file references: does file B exist and export what file A needs?
3. For HTML projects: are all JS files referenced via <script src="..."> tags?
   Context files provided: {ctx_list}
4. Duplicate or contradictory logic across files
{pattern_scan_hint}

Max 15 lines. Only report issues you can see in the actual code.
Format: "File: <n> | Issue: <one line>"
Mark cross-file issues [CROSS-FILE].
If everything looks correct: output "No issues found."
""".strip()

        scan = single_response(coder, scan_prompt, num_ctx=MAX_CTX_CODER, temperature=TEMPERATURE_CODING)
        emit("scan_done", scan)
    else:
        scan = "(No existing files — creating from scratch)"
        emit("status", "No files to scan — proceeding to plan...")

    # ------------------------------------------------------------------
    # STAGE 2: REASON  (constraint-rich planning)
    # ------------------------------------------------------------------
    emit("status", "Reasoning...")
    if not _swap_to(reasoner, emit, is_cancelled):
        return {}

    ctx_filenames = [os.path.basename(f) for f in (context_files or [])]
    if ctx_filenames:
        file_inventory = (
            "EXISTING FILES IN PROJECT (only modify these unless you must create a new one):\n"
            + "\n".join(f"  - {f}" for f in ctx_filenames)
            + "\n"
        )
    else:
        file_inventory = ""

    pattern_plan_hint = (
        f"\nKNOWN FAILURE PATTERNS TO GUARD AGAINST:\n{failure_patterns}\n"
        if failure_patterns else ""
    )

    history_hint = (
        f"\n{_history_block}\n" if _history_block else ""
    )

    reason_prompt = f"""
TASK: {task}{target_instruction}
{brief_section}
{file_inventory}SCAN FINDINGS: {scan}
{pattern_plan_hint}{history_hint}
Break this task into 3-5 steps. Each step = one substantial unit of work.
Aim for 50-150 lines of code per step.

CRITICAL FILE RULES:
- Prefer modifying existing files over creating new ones.
- If you create a new .js file, it MUST be <script src="..."> linked in the HTML in the same plan.
- Do NOT plan steps that write to files that will not be used anywhere.
- Consolidate logic: one HTML + one JS is better than one HTML + three JS files.

MOST IMPORTANT PLANNING RULE — FILE REWRITES ARE DANGEROUS:
Every time a step writes to a file, it must output the COMPLETE file from scratch.
This means each rewrite is an opportunity to accidentally lose code from a prior step.
To minimise this risk:
- Each file should be written by AS FEW STEPS AS POSSIBLE — ideally just one.
- If game.js needs state variables, movement, collision, food, and rendering, put ALL
  of that in a SINGLE step, not spread across 6 steps each rewriting the same file.
- It is FAR safer to have one large step writing a complete file than six small
  steps each partially rewriting it.
- Only split into multiple steps when the work genuinely belongs in separate files.

GOOD plan (2 steps, each file written once):
  STEP 1: Create index.html with full UI structure
  STEP 2: Create game.js with complete game logic including all state, movement, collision, rendering

BAD plan (dangerous, game.js rewritten 6 times):
  STEP 1: game.js — initialise variables
  STEP 2: game.js — add movement
  STEP 3: game.js — add collision
  STEP 4: game.js — add food
  STEP 5: game.js — add rendering
  STEP 6: game.js — add polish

After the step list, output these three sections EXACTLY as formatted:

CONSTRAINTS:
- <Specific, testable invariant the code must maintain at all times>
(List 3-6 constraints)

EDGE_CASES:
- <Scenario that might break the logic + expected correct behaviour>
(List 3-5 edge cases)

FAILURE_MODES:
- <Most common way this type of code breaks + how to prevent it>
(List 2-4 failure modes)

FRAMEWORK / LANGUAGE CONSTRAINTS:
Identify the exact language or framework the task requires and include its
critical rules in your CONSTRAINTS section. Do not hardcode generic rules —
derive them from what the task actually asks for. Examples:
- PySide6 task → include: "Use Signal not pyqtSignal", "Import from PySide6 not PyQt5", "Use app.exec() not exec_()"
- React task → include: "Use functional components with hooks, not class components"
- Java task → include: "Use standard Java 11+ APIs", "No raw types"
- C++ task → include: "Use smart pointers, avoid raw new/delete"
If the task does not specify a framework, do not invent framework constraints.

Use EXACTLY this format for EVERY step:

STEP 1: <one clear action sentence>
FILES: <filename.ext>
DEPENDS_ON: none
SUCCESS_CRITERIA: <one observable outcome>

STEP 2: <one clear action sentence>
FILES: <filename.ext>
DEPENDS_ON: STEP 1
SUCCESS_CRITERIA: <one observable outcome>

Output STEP blocks first, then CONSTRAINTS / EDGE_CASES / FAILURE_MODES.
Each step touches MINIMUM files needed.
SUCCESS_CRITERIA must be specific and observable.
""".strip()

    plan = single_response(reasoner, reason_prompt, num_ctx=MAX_CTX_REASONER, temperature=TEMPERATURE_PLANNING)
    emit("plan_done", plan)

    # ── SPEC VALIDATION: hard gate on planner output ─────────────────────
    # GPT: "If spec is not valid → retry. No exceptions. No 'good enough'."
    # We validate on up to MAX_PLAN_RETRIES attempts before continuing.
    # Checks (all must pass):
    #   1. Required sections present (CONSTRAINTS, EDGE_CASES, FAILURE_MODES)
    #   2. Each required section has at least one non-empty bullet/line
    #   3. At least one STEP block exists in the plan
    MAX_PLAN_RETRIES = 2
    for _plan_attempt in range(MAX_PLAN_RETRIES + 1):
        _spec_errors = _validate_plan_spec(plan)
        if not _spec_errors:
            break
        if _plan_attempt < MAX_PLAN_RETRIES:
            log(f"[pipeline] Plan spec invalid (attempt {_plan_attempt + 1}): {_spec_errors} — retrying planner")
            emit("status", f"Plan spec invalid — retrying planner... ({'; '.join(_spec_errors)})")
            _fix_prompt = (
                f"{reason_prompt}\n\n"
                f"CRITICAL: Your previous plan failed validation for these reasons:\n"
                + "\n".join(f"  - {e}" for e in _spec_errors) + "\n\n"
                f"Requirements that MUST be met:\n"
                f"  - Include CONSTRAINTS: section with at least 3 specific, testable constraints\n"
                f"  - Include EDGE_CASES: section with at least 2 edge cases\n"
                f"  - Include FAILURE_MODES: section with at least 2 failure modes\n"
                f"  - Include at least 1 STEP block in the format specified\n"
                f"Re-output the complete plan satisfying ALL requirements."
            )
            plan = single_response(reasoner, _fix_prompt, num_ctx=MAX_CTX_REASONER, temperature=TEMPERATURE_PLANNING)
            emit("plan_done", plan)
        else:
            log(f"[pipeline] Plan spec still invalid after {MAX_PLAN_RETRIES} retries — continuing with best available")

    plan_constraints = _extract_plan_section(plan, "CONSTRAINTS")
    plan_edge_cases  = _extract_plan_section(plan, "EDGE_CASES")
    plan_failures    = _extract_plan_section(plan, "FAILURE_MODES")
    constraints_block = _format_constraints_block(
        plan_constraints, plan_edge_cases, plan_failures
    )

    # ------------------------------------------------------------------
    # STAGE 3: DECOMPOSE
    # ------------------------------------------------------------------
    steps = parse_steps(plan)

    # ── ARTIFACT LOCK: freeze the spec ───────────────────────────────────
    # GPT: "Once spec is created: freeze it. All iterations operate against
    # that fixed spec. This removes one of the biggest sources of instability."
    # frozen_spec is immutable from this point — never regenerated.
    # Every downstream stage (coder, reviewer, retry) references this object.
    # NOTE: must come AFTER parse_steps so steps is defined.
    frozen_spec = _build_frozen_spec(plan, steps, constraints_block)
    log(f"[pipeline] Spec frozen: {len(steps)} steps, constraints={bool(plan_constraints)}")
    emit("status", f"Spec locked ({len(steps)} steps)")

    if not _swap_to(coder, emit, is_cancelled):
        return {}

    if not steps:
        log("[pipeline] No structured steps — falling back to single-step")
        emit("status", "Writing code (single pass)...")
        return _single_step_fallback(
            task, plan, file_context, brief_section,
            target_instruction, coder, emit, is_cancelled,
            project_dir, context_files, failure_patterns, constraints_block
        )

    emit("status", f"Plan: {len(steps)} steps")

    # ------------------------------------------------------------------
    # STAGE 4: STEP EXECUTION LOOP
    # ------------------------------------------------------------------
    state = StepState(
        project_dir=project_dir,
        task=task,
        steps=steps,
    )

    loop_result = run_steps(
        state             = state,
        task              = task,
        all_context_files = context_files or [],
        project_dir       = project_dir,
        brief_content     = brief_content,
        coder_model       = coder,
        stable_mode       = stable_mode,
        progress_callback = progress_callback,
        cancel_check      = cancel_check,
        constraints_block = constraints_block,
    )

    if is_cancelled():
        return {}

    completed_files = loop_result.get("completed_files", [])
    diffs           = loop_result.get("diffs", [])
    failed_steps    = loop_result.get("failed_steps", [])

    if diffs:
        emit("code_done", "Files written:\n" + "\n".join(diffs))

    if failed_steps:
        emit("status", f"{len(failed_steps)} step(s) skipped: {failed_steps}")

    # ------------------------------------------------------------------
    # STAGE 5: SIMULATE
    # ------------------------------------------------------------------
    if is_cancelled():
        return {}

    final_file_content = _collect_file_content(completed_files, project_dir, file_context)

    sim_result = {"skipped": True, "verdict": "PASS", "issues": [], "output": ""}
    if final_file_content and final_file_content.strip():
        emit("status", "Simulating execution...")
        sim_result = run_simulation(
            task             = task,
            file_content     = final_file_content,
            context_files    = context_files or [],
            coder_model      = coder,
            failure_patterns = failure_patterns,
        )
        sim_label = "ISSUES FOUND" if sim_result["verdict"] == "ISSUES_FOUND" else "PASS"
        emit("simulate_done", f"Simulation: {sim_label}\n\n{sim_result['output']}")
    else:
        emit("status", "Skipping simulation (no output to trace)")

    # ------------------------------------------------------------------
    # STAGE 6: FINAL REVIEW
    # ------------------------------------------------------------------
    if is_cancelled():
        return {}

    emit("status", "Final review...")

    written_filenames = [
        os.path.relpath(f, project_dir) if project_dir else f
        for f in completed_files if os.path.exists(f)
    ]
    written_list = "\n".join(f"  - {f}" for f in written_filenames) or "  (none)"

    simulation_note = ""
    if not sim_result.get("skipped") and sim_result["verdict"] == "ISSUES_FOUND":
        simulation_note = (
            "\nSIMULATION STAGE FINDINGS (address these):\n"
            + "\n".join(f"  - {i}" for i in sim_result["issues"])
            + "\n"
        )

    pattern_review_hint = (
        f"\nKNOWN FAILURE PATTERNS (verify NOT present):\n{failure_patterns}"
        if failure_patterns else ""
    )

    constraints_review_hint = (
        f"\nCONSTRAINTS FROM PLAN (verify all satisfied):\n{plan_constraints}"
        if plan_constraints else ""
    )

    review_prompt = f"""
You are a code reviewer. Your ONLY job is to compare the code against the spec and list mismatches.
Do NOT suggest general improvements. Do NOT rewrite working code. Fix ONLY what differs from the spec.

ORIGINAL TASK: {task}
SPEC (plan the code was supposed to follow — this is frozen and authoritative):
{extract_plan_summary(frozen_spec["steps"])}

FILES WRITTEN:
{written_list}

CODE OUTPUT:
{final_file_content}
{simulation_note}{pattern_review_hint}{constraints_review_hint}

REVIEW PROCESS — follow these steps in order:
1. For each CONSTRAINT in the spec: does the code satisfy it? List any that are violated.
2. For each STEP's SUCCESS_CRITERIA: is it met in the code? List any that are not met.
3. Check for mechanical errors only:
   - Wrong variable/function names vs spec
   - Missing or mismatched function signatures
   - Dead code (defined but never called)
   - Orphan .js files without a <script src="..."> in HTML
   - Cross-file interface mismatches (file A expects X, file B provides Y)
   - Obvious logic bugs (off-by-one, wrong condition, wrong operator)
4. For HTML/JS projects, check UI reachability:
   - Is every menu/screen the user needs actually VISIBLE when they open the page?
   - Any element with display:none that the user must see — is there code that shows it?
   - Does the intended first screen (start menu, login, etc.) actually appear on load?
   - Does game/app logic start ONLY after the user triggers it (click, etc.), not on page load?
   - If the task says "start menu", verify the start menu is visible and functional before gameplay begins.
5. Do NOT flag style, naming preferences, or "could be cleaner" — only SPEC VIOLATIONS and BUGS.

STOP CONDITION: If you find zero mismatches and zero bugs, output VERDICT: PASS.
Only output NEEDS_CHANGES if there are specific, concrete, fixable issues listed below.

VERDICT: PASS / NEEDS_CHANGES / FAIL
MISMATCHES: (number each one, or "None")
  1. <file>: <what spec says> vs <what code does>
  2. ...
FIX_ONLY: (list what to change — nothing else, or "None")
""".strip()

    review  = single_response(coder, review_prompt, num_ctx=MAX_CTX_CODER, temperature=TEMPERATURE_CODING)
    verdict = extract_verdict(review)
    emit("review_done", f"VERDICT: {verdict}\n\n{review}")

    # ------------------------------------------------------------------
    # STAGE 7: MULTI-PASS RETRY LOOP
    # ------------------------------------------------------------------
    final_code = final_file_content
    retry_pass = 0

    while verdict in ("NEEDS_CHANGES", "FAIL") and retry_pass < MAX_RETRY_PASSES:
        if is_cancelled():
            return {}

        retry_pass += 1
        emit("status", f"Fixing issues (pass {retry_pass}/{MAX_RETRY_PASSES})...")

        sim_issues_note = format_simulation_for_retry(sim_result)

        _junk = str.maketrans("", "", "`*_\"'")
        code_files = [f.strip().translate(_junk).strip(",:; ")
                      for f in re.findall(r"FILE:\s*([^\n]+)\n```", final_code)]
        file_list = "\n".join(f"- FILE: {f}" for f in code_files if f) if code_files \
                    else "- same files as in the previous code"

        retry_prompt = f"""
The previous code had spec mismatches. Fix ONLY those mismatches — do not change anything else.
Correction pass {retry_pass} of {MAX_RETRY_PASSES}.

ORIGINAL TASK: {task}{target_instruction}

MISMATCHES TO FIX (from reviewer — fix ALL of these, nothing more):
{review}

{sim_issues_note}
{constraints_block}

PLAN (spec the code must match — frozen and authoritative):
{extract_plan_summary(frozen_spec["steps"])}

PREVIOUS CODE:
{final_code}

Output ALL of the following files COMPLETE and CORRECTED:
{file_list}

FILE: filename.ext
```language
<complete file contents — every line, no omissions>
```

Rules:
- One FILE: block per file listed
- Complete file contents only — never partial or snippets
- No explanations outside FILE blocks
- Fix the listed mismatches ONLY — do not rename, restructure, or optimise anything else
- Every variable, function, and class name MUST stay identical to the previous version
  UNLESS the reviewer specifically flagged it as wrong
""".strip()

        retry_output = single_response(coder, retry_prompt, num_ctx=MAX_CTX_CODER, temperature=TEMPERATURE_CODING)

        import re as _re
        retry_count = len(_re.findall(r"FILE:\s*[^\n]+\n```", retry_output))
        code_count  = len(_re.findall(r"FILE:\s*[^\n]+\n```", final_code))

        if retry_count > 0 and retry_count >= code_count:
            final_code = retry_output
            emit("retry_done", f"Pass {retry_pass}:\n{retry_output}")

            # ── CRITICAL: write corrected files to disk ───────────────
            # The retry fixed code in memory but the old broken files are
            # still on disk. Parse FILE: blocks and overwrite each file
            # so the fixed version is what actually gets delivered.
            _write_retry_files(retry_output, project_dir, completed_files, emit)

            if retry_pass < MAX_RETRY_PASSES:
                re_review_prompt = f"""
Re-review after correction pass {retry_pass}.
Your ONLY job: check if the previously listed mismatches have been fixed.
Do NOT introduce new checks or general feedback.

ORIGINAL TASK: {task}
PREVIOUS MISMATCHES THAT WERE FIXED:
{review}

CORRECTED CODE:
{final_code}

For each mismatch from the previous review:
- Is it now fixed? (yes/no)
- If no, describe what's still wrong.

STOP CONDITION: If ALL previous mismatches are fixed → VERDICT: PASS
Only output NEEDS_CHANGES if specific mismatches from the list above remain unfixed.

VERDICT: PASS / NEEDS_CHANGES / FAIL
REMAINING_MISMATCHES: (list only unfixed items from previous review, or "None")
NEW_BUGS_INTRODUCED: (regressions only — not style, not suggestions — or "None")
""".strip()
                re_review = single_response(coder, re_review_prompt, num_ctx=MAX_CTX_CODER, temperature=TEMPERATURE_CODING)
                new_verdict = extract_verdict(re_review)

                # True stop condition: zero remaining mismatches
                # Check for explicit "None" in REMAINING_MISMATCHES section
                _remaining = re.search(
                    r"REMAINING_MISMATCHES\s*:\s*(.+?)(?:\n[A-Z_]+\s*:|$)",
                    re_review, re.DOTALL | re.IGNORECASE
                )
                _remaining_text = _remaining.group(1).strip() if _remaining else ""
                _zero_mismatches = (
                    _remaining_text.lower() in ("none", "none.", "") or
                    not re.search(r"\d+\.", _remaining_text)  # no numbered items
                )

                if _zero_mismatches or new_verdict == "PASS":
                    new_verdict = "PASS"
                    emit("status", f"✅ Zero mismatches — stopping at retry pass {retry_pass}")

                emit("review_done", f"Re-review pass {retry_pass}: VERDICT: {new_verdict}\n\n{re_review}")
                verdict = new_verdict
                review  = re_review
            else:
                verdict = "PASS"
        else:
            log(f"[pipeline] retry pass {retry_pass}: {retry_count} FILE: blocks vs {code_count} — keeping previous")
            emit("retry_done", f"(Pass {retry_pass} did not improve output — keeping previous)")
            break

    if completed_files and project_dir:
        update_summaries(project_dir, completed_files)

    if project_dir:
        rel_files = [os.path.relpath(f, project_dir) for f in completed_files if os.path.exists(f)]
        append_run_summary(project_dir, task, rel_files, verdict)

    # ── SANITY VALIDATION ────────────────────────────────────────────────
    # GPT fix #6: lightweight mechanical checks before declaring success.
    # Catches empty/partial outputs that passed the review stage regardless.
    sanity = _run_sanity_checks(completed_files, project_dir, task)
    if sanity["warnings"]:
        for w in sanity["warnings"]:
            emit("status", f"⚠ Sanity check: {w}")
        log(f"[pipeline] Sanity warnings: {sanity['warnings']}")
    if sanity["hard_fail"] and verdict == "PASS":
        verdict = "NEEDS_CHANGES"
        log(f"[pipeline] Sanity hard-fail overrode PASS verdict")

    log(f"[pipeline] done. verdict={verdict} files={len(completed_files)} retry_passes={retry_pass}")
    return {
        "scan":            scan,
        "plan":            plan,
        "steps":           steps,
        "review":          review,
        "verdict":         verdict,
        "final_code":      final_code,
        "completed_files": completed_files,
        "diffs":           diffs,
        "sim_result":      sim_result,
        "retry_passes":    retry_pass,
        "sanity":          sanity,
    }


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _collect_file_content(completed_files: list, project_dir: str, fallback: str) -> str:
    content = ""
    for fpath in completed_files:
        if os.path.exists(fpath):
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    text = f.read()
                rel = os.path.relpath(fpath, project_dir) if project_dir else fpath
                content += f"\nFILE: {rel}\n```\n{text}\n```\n"
            except Exception:
                pass
    return content or fallback


def _extract_plan_section(plan: str, section_name: str) -> str:
    pattern = rf"{section_name}[:\s]*\n(.*?)(?:\n[A-Z_]{{3,}}[:\s]*\n|\Z)"
    m = re.search(pattern, plan, re.DOTALL | re.IGNORECASE)
    if not m:
        return ""
    lines = [l.strip() for l in m.group(1).splitlines() if l.strip()]
    return "\n".join(lines)


def _format_constraints_block(constraints: str, edge_cases: str, failures: str) -> str:
    parts = []
    if constraints:
        parts.append(f"CONSTRAINTS:\n{constraints}")
    if edge_cases:
        parts.append(f"EDGE CASES:\n{edge_cases}")
    if failures:
        parts.append(f"FAILURE MODES:\n{failures}")
    return "\n\n".join(parts) if parts else ""


def _write_retry_files(
    retry_output: str,
    project_dir: str,
    completed_files: list,
    emit,
) -> None:
    """
    Parses FILE: blocks from a retry pass and writes them to disk.

    This is critical: the retry loop fixes code in memory and re-reviews it,
    but without writing to disk the original broken files remain. This function
    ensures the corrected version is what actually gets delivered to the user.

    Only writes files that were already written during the step loop
    (completed_files) — never creates new files from a retry pass.
    """
    from engine.apply_changes import extract_files
    from engine.step_state import StepState as _SS

    parsed = extract_files(retry_output)
    if not parsed:
        log("[pipeline] _write_retry_files: no FILE: blocks parsed — skipping disk write")
        return

    # Build a set of basenames already written so we don't create rogue files
    written_basenames = {
        os.path.basename(f).lower() for f in completed_files
    }

    for f in parsed:
        fname = os.path.basename(f["path"]).lower()
        if fname not in written_basenames:
            log(f"[pipeline] _write_retry_files: skipping unrecognised file '{f['path']}'")
            continue
        # Find the full path matching this filename
        full_path = next(
            (cf for cf in completed_files
             if os.path.basename(cf).lower() == fname),
            os.path.join(project_dir, f["path"]) if project_dir else f["path"]
        )
        try:
            os.makedirs(os.path.dirname(full_path) or ".", exist_ok=True)
            with open(full_path, "w", encoding="utf-8") as fh:
                fh.write(f["code"])
            log(f"[pipeline] _write_retry_files: wrote corrected {full_path}")
            emit("status", f"  ✓ Wrote corrected: {os.path.basename(full_path)}")
        except Exception as e:
            log(f"[pipeline] _write_retry_files: failed to write {full_path}: {e}")


def _single_step_fallback(
    task, plan, file_context, brief_section,
    target_instruction, coder, emit, is_cancelled,
    project_dir, context_files,
    failure_patterns="", constraints_block=""
) -> dict:
    emit("status", "Writing code...")

    files_section = "FILES:\n" + file_context if file_context else ""

    code_prompt = f"""
TASK: {task}{target_instruction}
EXECUTION PLAN:
{plan}

{files_section}

Execute the plan. Output EVERY changed or created file:

FILE: relative/path/to/file.ext
```language
<complete file contents here>
```

Always complete files — never partial. Correct language tag. No explanations outside FILE blocks.
""".strip()

    code = single_response(coder, code_prompt, num_ctx=MAX_CTX_CODER, temperature=TEMPERATURE_CODING)
    emit("code_done", code)

    sim_result = {"skipped": True, "verdict": "PASS", "issues": [], "output": ""}
    if code.strip():
        emit("status", "Simulating execution...")
        sim_result = run_simulation(
            task=task, file_content=code, context_files=context_files or [],
            coder_model=coder, failure_patterns=failure_patterns
        )
        sim_label = "ISSUES FOUND" if sim_result["verdict"] == "ISSUES_FOUND" else "PASS"
        emit("simulate_done", f"Simulation: {sim_label}\n\n{sim_result['output']}")

    simulation_note = ""
    if not sim_result.get("skipped") and sim_result["verdict"] == "ISSUES_FOUND":
        simulation_note = (
            "\nSIMULATION FINDINGS:\n"
            + "\n".join(f"  - {i}" for i in sim_result["issues"]) + "\n"
        )

    pattern_hint = (
        f"\nKNOWN FAILURE PATTERNS (verify NOT present):\n{failure_patterns}"
        if failure_patterns else ""
    )

    review_prompt = f"""
You are a code reviewer. Review this AI-generated code.

ORIGINAL TASK: {task}
PLAN: {plan}
CODE OUTPUT: {code}
{simulation_note}{pattern_hint}{constraints_block}

Check: correctness, logic errors, dead code, cross-file consistency,
function signatures, missing error handling, orphan JS files not in HTML.

VERDICT: PASS / NEEDS_CHANGES / FAIL
ISSUES: (list, or "None")
SUGGESTIONS: (specific improvements, or "None")
""".strip()

    review  = single_response(coder, review_prompt, num_ctx=MAX_CTX_CODER, temperature=TEMPERATURE_CODING)
    verdict = extract_verdict(review)
    emit("review_done", f"VERDICT: {verdict}\n\n{review}")

    final_code = code
    retry_pass = 0

    while verdict in ("NEEDS_CHANGES", "FAIL") and retry_pass < MAX_RETRY_PASSES:
        if is_cancelled():
            break

        retry_pass += 1
        emit("status", f"Fixing (pass {retry_pass}/{MAX_RETRY_PASSES})...")

        sim_issues_note = format_simulation_for_retry(sim_result)
        _junk = str.maketrans("", "", "`*_\"'")
        code_files = [f.strip().translate(_junk).strip(",:; ")
                      for f in re.findall(r"FILE:\s*([^\n]+)\n```", final_code)]
        file_list  = "\n".join(f"- FILE: {f}" for f in code_files if f) if code_files else ""

        retry_prompt = f"""
Fix reviewer issues. Output ALL files complete.
Pass {retry_pass}/{MAX_RETRY_PASSES}.

TASK: {task}{target_instruction}
FEEDBACK: {review}
{sim_issues_note}
{constraints_block}
PREVIOUS CODE: {final_code}
REQUIRED FILES: {file_list}

FILE: filename.ext
```language
<complete file>
```
""".strip()
        retry = single_response(coder, retry_prompt, num_ctx=MAX_CTX_CODER, temperature=TEMPERATURE_CODING)

        import re as _re
        if len(_re.findall(r"FILE:\s*[^\n]+\n```", retry)) > 0:
            final_code = retry
            emit("retry_done", f"Pass {retry_pass}:\n{final_code}")
            # Write corrected files to disk (same fix as main pipeline retry)
            _write_retry_files(final_code, project_dir or "", [], emit)

            if retry_pass < MAX_RETRY_PASSES:
                re_review = single_response(coder, f"""
Re-review after correction pass {retry_pass}.
TASK: {task}
PREVIOUS ISSUES: {review}
CORRECTED CODE: {final_code}
VERDICT: PASS / NEEDS_CHANGES / FAIL
ISSUES: (remaining/new, or "None")
SUGGESTIONS: (or "None")
""".strip(), num_ctx=MAX_CTX_CODER)
                verdict = extract_verdict(re_review)
                emit("review_done", f"Re-review pass {retry_pass}: VERDICT: {verdict}\n\n{re_review}")
                review = re_review
            else:
                verdict = "PASS"
        else:
            break

    return {
        "scan": "", "plan": plan, "steps": [],
        "review": review, "verdict": verdict,
        "final_code": final_code, "completed_files": [], "diffs": [],
        "sim_result": sim_result, "retry_passes": retry_pass,
    }


# --------------------------------------------------------------------------
# Refinement pass (outer loop pass 2+)
# --------------------------------------------------------------------------

def _run_refinement_pass(
    task:             str,
    frozen_spec:      dict,
    project_dir:      str,
    context_files:    list,
    coder:            str,
    emit,
    is_cancelled,
    failure_patterns: list,
    brief_section:    str,
) -> dict:
    """
    Pass 2+ of the outer loop.

    Does NOT re-plan or re-execute steps.
    Instead:
      1. Reads the current on-disk files as the baseline
      2. Runs the diff-based reviewer against the frozen spec
      3. If mismatches found, runs the retry loop to fix ONLY those
      4. Returns in the same result dict shape as run_pipeline

    This ensures each pass hones the existing output rather than
    rewriting from scratch.
    """
    from engine.apply_changes import extract_files, write_files

    # Read current on-disk files into context
    existing_content = ""
    written_files = []
    if project_dir and context_files:
        for fpath in context_files:
            if os.path.exists(fpath):
                written_files.append(fpath)
                try:
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as fh:
                        content = fh.read()
                    rel = os.path.relpath(fpath, project_dir)
                    existing_content += f"\nFILE: {rel}\n```\n{content}\n```\n"
                except Exception:
                    pass

    if not existing_content:
        log("[refinement] No existing files found on disk — nothing to refine")
        emit("status", "⚠ Refinement: no existing files to refine")
        return {
            "scan": "", "plan": frozen_spec.get("raw_plan", ""),
            "steps": frozen_spec.get("steps", []),
            "review": "", "verdict": "PASS",
            "final_code": "", "completed_files": [],
            "diffs": [], "sim_result": {}, "retry_passes": 0,
            "sanity": {"warnings": ["No files found for refinement"], "hard_fail": False},
        }

    steps      = frozen_spec.get("steps", [])
    constraints_block = frozen_spec.get("constraints", "")
    written_list = "\n".join(
        f"  - {os.path.relpath(f, project_dir)}" for f in written_files
    ) if written_files else "  (none)"

    # ── Diff-based review against frozen spec ────────────────────────
    emit("status", "Reviewing existing files against frozen spec...")
    review_prompt = f"""
You are a code reviewer doing a REFINEMENT pass.
The spec below is FROZEN — it is the authoritative reference. Do not question it.
Your job: find mismatches between the spec and the CURRENT CODE, and list them precisely.

ORIGINAL TASK: {task}

FROZEN SPEC:
{extract_plan_summary(steps)}

CONSTRAINTS:
{constraints_block}

CURRENT FILES:
{written_list}

CURRENT CODE:
{existing_content}

REVIEW PROCESS:
1. For each CONSTRAINT: does the code satisfy it? List violations.
2. For each STEP's SUCCESS_CRITERIA: is it met? List failures.
3. Check for mechanical bugs only (wrong names, missing calls, broken interfaces).
4. For HTML/JS projects, check UI reachability:
   - Is every screen the user needs actually visible when they open the page?
   - Any element with display:none that the user must see — is there code that shows it?
   - Does game/app logic start ONLY after the user triggers it, not on page load?
5. Do NOT suggest style changes, optimisations, or "could be better" items.

STOP CONDITION: If zero mismatches → VERDICT: PASS

VERDICT: PASS / NEEDS_CHANGES / FAIL
MISMATCHES:
  1. <file>: <what spec says> vs <what code does>
FIX_ONLY: (list changes needed, or "None")
""".strip()

    review  = single_response(coder, review_prompt, num_ctx=MAX_CTX_CODER, temperature=TEMPERATURE_CODING)
    verdict = extract_verdict(review)
    emit("review_done", f"Refinement review: VERDICT: {verdict}\n\n{review}")

    if verdict == "PASS" or is_cancelled():
        log("[refinement] Already PASS — no changes needed")
        sanity = _run_sanity_checks(written_files, project_dir, task)
        return {
            "scan": "", "plan": frozen_spec.get("raw_plan", ""),
            "steps": steps, "review": review, "verdict": verdict,
            "final_code": existing_content, "completed_files": written_files,
            "diffs": [], "sim_result": {}, "retry_passes": 0,
            "sanity": sanity,
        }

    # ── Targeted fix pass — fix ONLY listed mismatches ───────────────
    MAX_REFINEMENT_RETRIES = 3
    final_code  = existing_content
    retry_pass  = 0
    file_list   = "\n".join(
        f"  FILE: {os.path.relpath(f, project_dir)}" for f in written_files
    ) if written_files else ""

    for retry_pass in range(1, MAX_REFINEMENT_RETRIES + 1):
        if is_cancelled():
            break
        if verdict == "PASS":
            break

        emit("status", f"Fixing mismatches (refinement pass {retry_pass}/{MAX_REFINEMENT_RETRIES})...")

        fix_prompt = f"""
Fix ONLY the listed mismatches. Do NOT change anything else.
Every variable name, function name, and structure MUST stay identical unless
the mismatch explicitly requires renaming it.

ORIGINAL TASK: {task}
FROZEN SPEC:
{extract_plan_summary(steps)}

MISMATCHES TO FIX:
{review}

CURRENT CODE (baseline — preserve everything not in the mismatch list):
{final_code}

Output ALL files COMPLETE:
{file_list}

FILE: filename.ext
```language
<complete corrected file>
```
""".strip()

        retry_output = single_response(coder, fix_prompt, num_ctx=MAX_CTX_CODER, temperature=TEMPERATURE_CODING)
        emit("retry_done", f"Refinement fix pass {retry_pass}:\n{retry_output[:200]}...")

        parsed = extract_files(retry_output, task=task)
        if parsed and project_dir:
            written = write_files(parsed, project_dir)
            for dest in written:
                if dest not in written_files:
                    written_files.append(dest)
                log(f"[refinement] Wrote: {dest}")
                emit("status", f"  ✓ Refined: {os.path.basename(dest)}")
            final_code = retry_output

        # Re-review
        re_review_prompt = f"""
Re-review after refinement fix pass {retry_pass}.
Check ONLY whether the previously listed mismatches are now resolved.

PREVIOUS MISMATCHES:
{review}

UPDATED CODE:
{final_code}

VERDICT: PASS / NEEDS_CHANGES / FAIL
REMAINING_MISMATCHES: (list unfixed items, or "None")
NEW_BUGS_INTRODUCED: (regressions only, or "None")
""".strip()

        re_review = single_response(coder, re_review_prompt, num_ctx=MAX_CTX_CODER, temperature=TEMPERATURE_CODING)
        new_verdict = extract_verdict(re_review)

        _remaining = re.search(
            r"REMAINING_MISMATCHES\s*:\s*(.+?)(?:\n[A-Z_]+\s*:|$)",
            re_review, re.DOTALL | re.IGNORECASE
        )
        _remaining_text = _remaining.group(1).strip() if _remaining else ""
        _zero_mismatches = (
            _remaining_text.lower() in ("none", "none.", "") or
            not re.search(r"\d+\.", _remaining_text)
        )
        if _zero_mismatches or new_verdict == "PASS":
            new_verdict = "PASS"
            emit("status", f"✅ Zero mismatches — refinement complete at pass {retry_pass}")

        emit("review_done", f"Refinement re-review pass {retry_pass}: VERDICT: {new_verdict}\n\n{re_review}")
        verdict = new_verdict
        review  = re_review

    sanity = _run_sanity_checks(written_files, project_dir, task)
    return {
        "scan": "", "plan": frozen_spec.get("raw_plan", ""),
        "steps": steps, "review": review, "verdict": verdict,
        "final_code": final_code, "completed_files": written_files,
        "diffs": [f"[REFINED] {os.path.basename(f)}" for f in written_files],
        "sim_result": {}, "retry_passes": retry_pass,
        "sanity": sanity,
    }


# --------------------------------------------------------------------------
# Frozen spec builder (artifact lock)
# --------------------------------------------------------------------------

def _build_frozen_spec(plan: str, steps: list, constraints_block: str) -> dict:
    """
    Build an immutable spec object from the planner output.
    This is created once and referenced by all downstream stages.
    Downstream stages must NOT regenerate or modify the spec — only read it.
    """
    return {
        "raw_plan":         plan,
        "steps":            steps,
        "constraints":      constraints_block,
        "step_count":       len(steps),
        "step_descriptions": [s.get("description", "") for s in steps],
        "step_files":       [s.get("files", []) for s in steps],
        "success_criteria": [s.get("success_criteria", "") for s in steps],
    }


# --------------------------------------------------------------------------
# Plan spec validator
# --------------------------------------------------------------------------

def _validate_plan_spec(plan: str) -> list:
    """
    Hard gate: returns a list of error strings if the plan is invalid.
    Empty list = plan is valid and may proceed to DECOMPOSE.

    Rules (all must pass):
      1. CONSTRAINTS section exists and has >= 1 non-empty content line
      2. EDGE_CASES section exists and has >= 1 non-empty content line
      3. FAILURE_MODES section exists and has >= 1 non-empty content line
      4. At least one STEP block (STEP N:) exists

    "Content line" is deliberately broad: accepts bullets (- / *), numbered
    items (1.), bold markdown (**text**), or any non-empty line.
    The validator rejects only completely *empty* sections.
    """
    errors = []
    upper = plan.upper()

    def _section_has_content(section_name: str) -> bool:
        """Return True if the named section has at least one non-empty line."""
        content = _extract_plan_section(plan, section_name)
        if content.strip():
            return True
        # Fallback: search for the header and grab lines after it
        pattern = rf"{section_name}\s*[:\*]*\s*\n(.*?)(?:\n#{1,4}\s|\n[A-Z_]{{3,}}\s*[:\*]|\Z)"
        m = re.search(pattern, plan, re.DOTALL | re.IGNORECASE)
        if m:
            lines = [l.strip() for l in m.group(1).splitlines()
                     if l.strip() and not l.strip().startswith("```")]
            return bool(lines)
        return False

    for section in ("CONSTRAINTS", "EDGE_CASES", "FAILURE_MODES"):
        if section not in upper:
            errors.append(f"Missing {section} section")
        elif not _section_has_content(section):
            errors.append(f"{section} section is empty (needs at least one item)")

    # Accept "STEP 1:", "STEP 1 —", "#### STEP 1:", "**STEP 1**"
    if not re.search(r"STEP\s+\d+", plan, re.IGNORECASE):
        errors.append("No STEP blocks found — plan must contain at least one STEP N: block")

    return errors


# --------------------------------------------------------------------------
# Sanity validator
# --------------------------------------------------------------------------

def _run_sanity_checks(completed_files: list, project_dir: str, task: str) -> dict:
    """
    Lightweight mechanical checks on pipeline output — catches empty/partial
    outputs that slip through review. Returns warnings and a hard_fail flag.

    Checks:
      1. At least one file was written
      2. Each written file exists on disk and is non-empty (>= 5 lines)
      3. Any .js file referenced in HTML actually has a FILE: entry written
      4. No file contains obvious truncation markers ("# ... rest of file")
    """
    warnings = []
    hard_fail = False

    if not completed_files:
        warnings.append("No files were written — pipeline produced no output")
        hard_fail = True
        return {"warnings": warnings, "hard_fail": hard_fail}

    MIN_LINES = 5
    html_files = []
    js_files_written = set()

    for fpath in completed_files:
        if not os.path.exists(fpath):
            warnings.append(f"Expected output file missing from disk: {os.path.basename(fpath)}")
            hard_fail = True
            continue

        try:
            with open(fpath, "r", encoding="utf-8", errors="ignore") as fh:
                content = fh.read()
        except Exception:
            warnings.append(f"Could not read output file: {os.path.basename(fpath)}")
            continue

        line_count = content.count("\n") + 1
        if line_count < MIN_LINES:
            warnings.append(
                f"{os.path.basename(fpath)} is suspiciously short "
                f"({line_count} lines) — may be empty/partial"
            )
            hard_fail = True

        # Detect common truncation markers
        truncation_markers = [
            "# ... rest of", "# rest of the", "// ... rest",
            "<!-- rest of", "# (rest of", "// (rest of",
            "...existing code...", "/* rest of",
        ]
        lower = content.lower()
        for marker in truncation_markers:
            if marker in lower:
                warnings.append(
                    f"{os.path.basename(fpath)} contains truncation marker '{marker}' "
                    f"— file is incomplete"
                )
                hard_fail = True
                break

        ext = os.path.splitext(fpath)[1].lower()
        if ext == ".html":
            html_files.append((fpath, content))
        elif ext == ".js":
            js_files_written.add(os.path.basename(fpath).lower())

    # Check orphan JS: any .js <script src> in HTML that wasn't written
    for html_path, html_content in html_files:
        script_refs = re.findall(r'<script[^>]+src=["\']([^"\']+\.js)["\']', html_content, re.IGNORECASE)
        for ref in script_refs:
            ref_base = os.path.basename(ref).lower()
            # Only flag if it looks like a local file (no http/https)
            if not ref.startswith("http") and ref_base not in js_files_written:
                warnings.append(
                    f"HTML references {ref_base} but it was not written — "
                    f"page will have a missing script error"
                )
                # Soft warning only — don't hard-fail for this (might be pre-existing)

    # Python syntax check — catches syntax errors before the user runs the code
    if PYTHON_IMPORT_CHECK:
        import subprocess
        for fpath in completed_files:
            if not fpath.endswith(".py") or not os.path.exists(fpath):
                continue
            try:
                result = subprocess.run(
                    ["python3", "-m", "py_compile", fpath],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode != 0:
                    err = (result.stderr.strip().split("\n") or ["unknown error"])[-1]
                    warnings.append(
                        f"{os.path.basename(fpath)} has a syntax error: {err}"
                    )
                    hard_fail = True
            except Exception as e:
                log(f"[sanity] py_compile check failed for {fpath}: {e}")

    return {"warnings": warnings, "hard_fail": hard_fail}
