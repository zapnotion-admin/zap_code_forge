"""
engine/outer_loop.py
Outer convergence loop — wraps run_pipeline with a quality-judge.

The idea (from user brief):
  After each full pipeline run (SCAN → REASON → CODE → SIMULATE → REVIEW),
  ask a dedicated quality-judge: "Is this good enough?"
  If NO  → re-run the FULL pipeline from scratch using the current on-disk
            files as the new context, carrying forward what went wrong.
  If YES → stop. Return the result.

This is fundamentally different from the inner retry loop (which fixes
reviewer issues within one run). The outer loop is for cases where:
  - The generated code passed review but is still subtly wrong
  - A fresh reasoning pass with knowledge of the previous failure
    produces a better plan than patching the existing code
  - The problem is architectural (wrong approach) not just buggy

VRAM strategy:
  - Between iterations, the model is unloaded (VRAM freed to ~0)
  - File context is read fresh from disk each iteration
  - History/notes accumulated across iterations are stored in ContextCache
    (disk-backed if CONTEXT_CACHE_ENABLED) so they don't sit in VRAM

Quality judge:
  - Runs on the coder model (already loaded for the final review)
  - Uses a focused prompt: task + final code + reviewer notes
  - Returns GOOD_ENOUGH / NEEDS_IMPROVEMENT + a one-sentence reason
  - Deliberately conservative — "GOOD_ENOUGH" requires meeting ALL criteria
"""

import os
import re
import time
from engine.logger import log
from engine.ollama_client import single_response, unload_model, ensure_model, resolve_model
from engine.context_cache import ContextCache
from engine.context_manager import read_project_files
from core.config import (
    MODEL_CODER, MODEL_REASONER, MODEL_FALLBACK,
    MAX_CTX_CODER, MAX_OUTER_ITERATIONS, CONTEXT_CACHE_ENABLED,
    TEMPERATURE_CODING,
)


# ── Quality judge ─────────────────────────────────────────────────────────

def run_quality_judge(
    task:        str,
    final_code:  str,
    review:      str,
    history:     str,
    coder_model: str,
    iteration:   int,
    max_iter:    int,
) -> dict:
    """
    Ask the model: is the current output good enough to stop?

    Returns:
        {
            "verdict":  "GOOD_ENOUGH" | "NEEDS_IMPROVEMENT",
            "reason":   <one sentence explanation>,
            "raw":      <full model output>
        }
    """
    # On the last iteration, always accept — don't loop forever
    if iteration >= max_iter:
        return {
            "verdict": "GOOD_ENOUGH",
            "reason":  f"Reached maximum iterations ({max_iter}) — accepting current output.",
            "raw":     "(max iterations reached)",
        }

    prompt = f"""
You are a quality judge for an AI coding agent. Your job is to decide whether
the generated code is good enough to deliver to the user, or whether another
full reasoning and coding pass would meaningfully improve it.

ORIGINAL TASK:
{task}

REVIEWER NOTES FROM THIS PASS:
{review}

{history}FINAL CODE OUTPUT:
{final_code[:3000]}
{"...(truncated)" if len(final_code) > 3000 else ""}

JUDGE CRITERIA — answer GOOD_ENOUGH only if ALL of the following are true:
1. The code correctly implements what the task asks for
2. No logic errors, state bugs, or timing issues remain
3. The reviewer gave PASS (or NEEDS_CHANGES that were fully resolved)
4. Another pass is unlikely to produce a meaningfully better result

Answer NEEDS_IMPROVEMENT if:
- There are known unfixed issues from the reviewer
- The code has structural problems that patching won't fix
- A fresh reasoning pass with a different approach would likely do better

Respond in EXACTLY this format (two lines only):
JUDGE_VERDICT: GOOD_ENOUGH or NEEDS_IMPROVEMENT
JUDGE_REASON: <one sentence — be specific about what is or isn't good enough>
""".strip()

    raw = single_response(coder_model, prompt, num_ctx=MAX_CTX_CODER, temperature=TEMPERATURE_CODING)

    verdict_match = re.search(r"JUDGE_VERDICT\s*:\s*(GOOD_ENOUGH|NEEDS_IMPROVEMENT)", raw, re.IGNORECASE)
    reason_match  = re.search(r"JUDGE_REASON\s*:\s*(.+)", raw)

    verdict = verdict_match.group(1).upper() if verdict_match else "GOOD_ENOUGH"
    reason  = reason_match.group(1).strip()  if reason_match  else raw.strip()[:200]

    log(f"[outer_loop] Judge iter={iteration}: {verdict} — {reason[:80]}")
    return {"verdict": verdict, "reason": reason, "raw": raw}


# ── Outer loop ────────────────────────────────────────────────────────────

def run_outer_loop(
    task:              str,
    file_context:      str,
    project_dir:       str = "",
    context_files:     list = None,
    coder_model:       str = None,
    reasoner_model:    str = None,
    stable_mode:       bool = True,
    max_iterations:    int = None,
    progress_callback  = None,
    cancel_check:      object = None,
) -> dict:
    """
    Outer convergence loop. Calls run_pipeline up to max_iterations times.
    After each run, a quality judge decides whether to stop or go again.

    On re-runs:
      - File context is re-read from disk (picks up whatever was written)
      - Iteration history is injected into the planner prompt
      - The task is annotated with what went wrong last time

    Parameters
    ----------
    max_iterations : int or None
        How many full pipeline passes to allow. None = use config default.
        1 = original single-pass behaviour (no outer loop).
    """
    from engine.workflow import run_pipeline

    coder    = resolve_model(coder_model    or MODEL_CODER,    MODEL_FALLBACK)
    reasoner = resolve_model(reasoner_model or MODEL_REASONER, MODEL_FALLBACK)

    max_iter = max_iterations if max_iterations is not None else MAX_OUTER_ITERATIONS
    max_iter = max(1, min(max_iter, 5))  # clamp 1–5

    def emit(stage: str, text: str) -> None:
        log(f"[outer_loop] {stage}: {text[:80]}")
        if progress_callback:
            progress_callback(stage, text)

    def is_cancelled() -> bool:
        return cancel_check is not None and cancel_check()

    cache      = ContextCache(run_id=f"task_{int(time.time())}")
    result     = {}
    current_fc = file_context   # file_context for this iteration
    task_note  = ""             # carries forward what went wrong
    frozen_spec_from_pass1 = None   # set after pass 1, reused in all refinement passes

    try:
        for iteration in range(1, max_iter + 1):
            if is_cancelled():
                break

            if max_iter > 1:
                emit("status", f"━━ Outer loop: pass {iteration}/{max_iter} ━━")

            # Pass 1: full pipeline (scan → plan → execute → review)
            # Pass 2+: refinement mode (review + fix only, no replanning)
            is_refinement = (iteration > 1 and frozen_spec_from_pass1 is not None)

            if is_refinement:
                emit("status", f"[Refinement] Honing existing files — no replanning")
            else:
                # Only annotate task with error context on a fresh pass (not refinement)
                if task_note:
                    annotated_task = (
                        f"{task}\n\n"
                        f"[PREVIOUS ATTEMPT — iteration {iteration}]\n"
                        f"{task_note}\n"
                    )
                else:
                    annotated_task = task

            history_block = cache.format_history_for_prompt()

            result = run_pipeline(
                task              = task if is_refinement else annotated_task,
                file_context      = current_fc,
                project_dir       = project_dir,
                context_files     = context_files or [],
                coder_model       = coder,
                reasoner_model    = reasoner,
                stable_mode       = stable_mode,
                progress_callback = progress_callback,
                cancel_check      = cancel_check,
                _history_block    = history_block,
                _refinement_mode  = is_refinement,
                _frozen_spec      = frozen_spec_from_pass1 if is_refinement else None,
            )

            # Capture frozen spec after pass 1 so all later passes can use it
            if iteration == 1 and result.get("steps"):
                from engine.workflow import _build_frozen_spec
                frozen_spec_from_pass1 = _build_frozen_spec(
                    result.get("plan", ""),
                    result.get("steps", []),
                    "",  # constraints already embedded in steps
                )
                log(f"[outer_loop] Frozen spec captured: {len(result.get('steps', []))} steps")

            if is_cancelled() or not result:
                break

            # Single pass — no outer loop needed
            if max_iter == 1:
                break

            # ── Quality judge ────────────────────────────────────────
            verdict      = result.get("verdict", "PASS")
            review       = result.get("review", "")
            final_code   = result.get("final_code", "")
            sim_result   = result.get("sim_result", {})
            sim_verdict  = sim_result.get("verdict", "PASS") if sim_result else "PASS"
            sanity       = result.get("sanity", {})

            # Hard override: if sanity check failed (e.g. no files written,
            # truncated output), force NEEDS_IMPROVEMENT regardless of what
            # the judge or reviewer concluded. Prevents the contradiction of
            # "GOOD_ENOUGH" on a run that produced nothing.
            sanity_hard_fail = sanity.get("hard_fail", False)
            if sanity_hard_fail and iteration < max_iter:
                emit("status", "⚠ Sanity hard-fail — skipping quality judge, forcing next pass")
                log(f"[outer_loop] Sanity hard-fail on pass {iteration} — forcing NEEDS_IMPROVEMENT")
                cache.push_iteration(
                    iteration     = iteration,
                    plan          = result.get("plan", ""),
                    verdict       = verdict,
                    issues        = _extract_issues(review),
                    judge_verdict = "NEEDS_IMPROVEMENT",
                    judge_reason  = f"Sanity hard-fail: {'; '.join(sanity.get('warnings', []))}",
                    sim_verdict   = sim_verdict,
                )
                task_note = _build_task_note(review, sanity.get("warnings", []), sim_result)
                history_block = cache.format_history()
                iteration += 1
                unload_model()
                continue

            emit("status", f"🔍 Quality judge (pass {iteration}/{max_iter})...")

            judge = run_quality_judge(
                task        = task,
                final_code  = final_code,
                review      = review,
                history     = history_block,
                coder_model = coder,
                iteration   = iteration,
                max_iter    = max_iter,
            )

            emit(
                "judge_done",
                f"Pass {iteration}/{max_iter}: {judge['verdict']}\n{judge['reason']}"
            )

            # Record this iteration in the cache
            cache.push_iteration(
                iteration     = iteration,
                plan          = result.get("plan", ""),
                verdict       = verdict,
                issues        = _extract_issues(review),
                judge_verdict = judge["verdict"],
                judge_reason  = judge["reason"],
                sim_verdict   = sim_verdict,
            )

            # Stop if good enough
            if judge["verdict"] == "GOOD_ENOUGH":
                if max_iter > 1:
                    emit("status", f"✅ Quality judge: GOOD_ENOUGH — stopping at pass {iteration}")
                break

            # ── Prepare next iteration ───────────────────────────────
            # Unload model to free VRAM between passes
            unload_model()

            # Build task_note from issues for next pass annotation
            task_note = _build_task_note(review, judge["reason"], sim_result)

            # Re-read files from disk — picks up whatever was written this pass
            if project_dir and context_files:
                current_fc = _reload_file_context(project_dir, context_files)
                emit("status", f"📂 Re-read {len(context_files)} file(s) from disk for pass {iteration + 1}")

    finally:
        cache.close()

    result["outer_iterations"] = iteration if result else 0
    return result


# ── Helpers ───────────────────────────────────────────────────────────────

def _extract_issues(review: str) -> str:
    """Pull the ISSUES section from a review block."""
    m = re.search(r"ISSUES\s*:\s*\n(.*?)(?:\nSUGGESTIONS|\Z)", review, re.DOTALL | re.IGNORECASE)
    if not m:
        return ""
    lines = [l.strip() for l in m.group(1).splitlines() if l.strip() and l.strip().lower() != "none"]
    return "; ".join(lines[:5])  # cap at 5 issues for brevity


def _build_task_note(review: str, judge_reason: str, sim_result: dict) -> str:
    """Summarise what went wrong this pass for the next iteration's task annotation."""
    parts = []

    issues = _extract_issues(review)
    if issues:
        parts.append(f"Reviewer issues in previous pass: {issues}")

    if judge_reason and "GOOD_ENOUGH" not in judge_reason.upper():
        parts.append(f"Quality judge noted: {judge_reason}")

    if sim_result and not sim_result.get("skipped"):
        sim_issues = sim_result.get("issues", [])
        if sim_issues:
            parts.append(f"Simulation found: {'; '.join(sim_issues[:3])}")

    if not parts:
        parts.append("Previous pass did not fully satisfy the task requirements.")

    return "\n".join(parts)


def _reload_file_context(project_dir: str, context_files: list) -> str:
    """Re-read context files from disk and return as a FILE: block string."""
    filenames = [os.path.basename(f) for f in context_files]
    file_data = read_project_files(project_dir, filenames)

    parts = []
    for name, content in file_data.items():
        if content:
            parts.append(f"FILE: {name}\n```\n{content}\n```")
    return "\n\n".join(parts)
