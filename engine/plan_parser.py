"""
engine/plan_parser.py
Parses the structured step plan produced by the REASON stage.

Expected format (must match what workflow.py prompts for):

STEP 1: <description>
FILES: <filename>
DEPENDS_ON: none
SUCCESS_CRITERIA: <criteria>

STEP 2: <description>
FILES: <filename>
DEPENDS_ON: STEP 1
SUCCESS_CRITERIA: <criteria>

v2 fixes:
- SUCCESS_CRITERIA parsing made more robust (handles missing field gracefully)
- extract_plan_summary now includes success criteria in the summary shown to executor
- steps_to_status_summary handles all step statuses correctly
"""

import re
from engine.logger import log


# Characters that LLMs commonly wrap filenames with in markdown output.
# e.g. `game.js`, **game.js**, *game.js*, "game.js", 'game.js'
_FILENAME_JUNK = str.maketrans("", "", "`*_\"'")


def _clean_filename(raw: str) -> str:
    """
    Strip markdown decoration from a filename the model emitted.

    Models frequently wrap filenames in backticks, bold markers, or quotes:
        `game.js`   -> game.js
        **game.js** -> game.js
        "index.html" -> index.html
        `index.html`, -> index.html   (trailing comma from list)

    Also strips surrounding whitespace and any trailing punctuation that
    isn't part of a valid filename (commas, colons, semicolons).
    """
    s = raw.strip()
    s = s.translate(_FILENAME_JUNK)   # remove backtick, *, _, quotes
    s = s.strip(",:;")                # remove trailing punctuation
    s = s.strip()
    return s


def parse_steps(plan_text: str) -> list:
    """
    Parse the REASON output into a list of step dicts.

    Each step dict:
      number:           int
      description:      str
      files:            list[str]
      depends_on:       list[str]   (e.g. ["STEP 1", "STEP 2"])
      success_criteria: str
      status:           "pending" | "in_progress" | "complete" | "failed"
    """
    # Strip code fences BEFORE splitting on STEP boundaries.
    # Planners sometimes embed example code inside the plan with comments like
    # "// game.js (Step 1)" — without stripping these, the regex below would
    # create phantom steps from those embedded comments.
    clean_plan = re.sub(r"```.*?```", "", plan_text, flags=re.DOTALL)

    # Also strip inline backtick spans that could contain "STEP N:" text
    clean_plan = re.sub(r"`[^`\n]+`", "", clean_plan)

    # Split on STEP N: boundaries
    blocks = re.split(r"(?=STEP\s+\d+\s*:)", clean_plan.strip(), flags=re.IGNORECASE)
    steps = []

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        # Extract step number and description from first line
        header = re.match(r"STEP\s+(\d+)\s*:\s*(.+)", block, re.IGNORECASE)
        if not header:
            continue

        number      = int(header.group(1))
        description = header.group(2).strip()

        # FILES:
        files_match = re.search(r"FILES?\s*:\s*(.+)", block, re.IGNORECASE)
        files_raw   = files_match.group(1).strip() if files_match else ""
        # Handle comma-separated or space-separated file lists.
        # Strip markdown decoration (backticks, bold, quotes) from each name.
        files = [
            _clean_filename(f)
            for f in re.split(r"[,\s]+", files_raw)
            if f.strip() and "." in f
        ]
        # Drop any that became empty after cleaning
        files = [f for f in files if f]

        # DEPENDS_ON:
        dep_match  = re.search(r"DEPENDS_ON\s*:\s*(.+)", block, re.IGNORECASE)
        dep_raw    = dep_match.group(1).strip() if dep_match else "none"
        if dep_raw.lower() in ("none", "n/a", "-", ""):
            depends_on = []
        else:
            depends_on = [d.strip().upper() for d in re.split(r"[,\s]+", dep_raw) if d.strip()]

        # SUCCESS_CRITERIA: (optional — missing field is fine)
        sc_match         = re.search(r"SUCCESS_CRITERIA\s*:\s*(.+)", block, re.IGNORECASE)
        success_criteria = sc_match.group(1).strip() if sc_match else ""

        steps.append({
            "number":           number,
            "description":      description,
            "files":            files,
            "depends_on":       depends_on,
            "success_criteria": success_criteria,
            "status":           "pending",
        })

    # Sort by step number (in case the model output them out of order)
    steps.sort(key=lambda s: s["number"])

    # Deduplicate by step number — keep first occurrence.
    seen = set()
    deduped = []
    for s in steps:
        if s["number"] not in seen:
            seen.add(s["number"])
            deduped.append(s)

    # Filter out non-executable steps — steps with no FILES field and
    # descriptions that are clearly non-coding (test, polish, sound, optimize).
    # These steps have nothing to write to disk and burn all retries on every pass.
    # Filter out non-executable steps.
    # A step is only executable if it has at least one file to write.
    # Steps with no FILES field (e.g. "Add polish", "Test", "Add game loop")
    # have nothing to write to disk — they burn all retries and produce nothing.
    # Any legitimate implementation step MUST name the files it will create/modify.
    executable = [s for s in deduped if s.get("files")]
    non_exec = [s["number"] for s in deduped if not s.get("files")]
    if non_exec:
        log(f"[plan_parser] Filtered {len(non_exec)} step(s) with no FILES field: {non_exec}")

    return executable


def extract_plan_summary(steps: list) -> str:
    """
    Compact summary of the full plan — injected into every step prompt
    so the model knows the big picture while executing one step.
    Includes success criteria so executor knows what each step should achieve.
    """
    if not steps:
        return "(no structured plan)"

    lines = []
    for s in steps:
        status_icon = {
            "pending":     "○",
            "in_progress": "▶",
            "complete":    "✓",
            "failed":      "✗",
        }.get(s.get("status", "pending"), "○")

        line = f"  {status_icon} Step {s['number']}: {s['description']}"
        if s.get("files"):
            line += f"  [{', '.join(s['files'])}]"
        if s.get("success_criteria"):
            line += f"\n       → {s['success_criteria']}"
        lines.append(line)

    return "\n".join(lines)


def steps_to_status_summary(steps: list) -> str:
    """
    One-liner per step showing current status.
    Shown in every step prompt so the model knows what's already done.
    """
    lines = []
    for s in steps:
        status = s.get("status", "pending")
        icon = {
            "pending":     "○ PENDING",
            "in_progress": "▶ IN PROGRESS",
            "complete":    "✓ DONE",
            "failed":      "✗ FAILED (skipped)",
        }.get(status, "○ PENDING")
        lines.append(f"  Step {s['number']}: {icon} — {s['description']}")
    return "\n".join(lines)
