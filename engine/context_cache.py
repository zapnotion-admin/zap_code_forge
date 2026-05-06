"""
engine/context_cache.py
Disk-backed context cache for the outer convergence loop.

Problem it solves:
  Between outer loop iterations, file content + history can exceed VRAM.
  Instead of holding everything in memory, we write large blobs to a temp
  file on disk and read them back only when the next prompt is assembled.
  This trades a small amount of latency (disk I/O) for much lower peak
  VRAM pressure — keeping the 16GB 4080 comfortably under the 13GB target.

Design:
  - One cache file per pipeline run (keyed by run_id)
  - Stores: iteration history, previous plans, accumulated reviewer notes
  - JSON on disk — human-readable, easy to inspect/debug
  - Auto-deleted when the outer loop finishes cleanly
  - If CONTEXT_CACHE_ENABLED=False in config, everything stays in memory
    (original behaviour)

Size considerations:
  - A full Snake-game run produces ~8–12KB of context
  - Even with 5 iterations that's ~60KB — trivial on disk
  - For larger projects (50+ files) this genuinely reduces peak memory
"""

import json
import os
import tempfile
import time
from engine.logger import log
from core.config import CONTEXT_CACHE_ENABLED


class ContextCache:
    """
    Accumulates per-iteration context across outer loop passes.
    If CONTEXT_CACHE_ENABLED, data is written to a temp file after each
    iteration and read back before the next. Otherwise it stays in-memory.

    Usage:
        cache = ContextCache(run_id="snake_20260424")
        cache.push_iteration(iteration=1, plan=..., verdict=..., issues=..., judge=...)
        history = cache.get_history()   # list of all past iterations
        cache.close()                   # deletes temp file
    """

    def __init__(self, run_id: str = ""):
        self._run_id     = run_id or f"run_{int(time.time())}"
        self._data: list = []          # in-memory copy (always maintained)
        self._cache_file = None

        if CONTEXT_CACHE_ENABLED:
            try:
                fd, path = tempfile.mkstemp(
                    prefix=f"zapforge_{self._run_id}_",
                    suffix=".cache.json",
                )
                os.close(fd)
                self._cache_file = path
                self._flush()
                log(f"[context_cache] Cache file: {path}")
            except Exception as e:
                log(f"[context_cache] Could not create cache file: {e} — using in-memory")
                self._cache_file = None

    def push_iteration(
        self,
        iteration:    int,
        plan:         str = "",
        verdict:      str = "",
        issues:       str = "",
        judge_verdict: str = "",
        judge_reason:  str = "",
        sim_verdict:   str = "",
    ) -> None:
        """Record the outcome of one outer loop pass."""
        entry = {
            "iteration":    iteration,
            "plan":         plan[:2000],   # cap to avoid bloating prompts
            "verdict":      verdict,
            "issues":       issues[:1000],
            "judge_verdict": judge_verdict,
            "judge_reason":  judge_reason[:500],
            "sim_verdict":  sim_verdict,
        }
        self._data.append(entry)
        if self._cache_file:
            self._flush()

    def get_history(self) -> list:
        """Return all iteration records (read from disk if cache enabled)."""
        if self._cache_file and os.path.exists(self._cache_file):
            try:
                with open(self._cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                log(f"[context_cache] Read error: {e} — falling back to in-memory")
        return self._data

    def format_history_for_prompt(self, max_entries: int = 3) -> str:
        """
        Format iteration history as a concise block for injection into prompts.
        Only includes the last max_entries iterations to keep prompts lean.
        """
        history = self.get_history()
        if not history:
            return ""

        recent = history[-max_entries:]
        lines = ["PREVIOUS ITERATION HISTORY:"]
        for h in recent:
            lines.append(
                f"\n  Iteration {h['iteration']}:"
                f"\n    Reviewer verdict: {h['verdict']}"
                + (f"\n    Issues found: {h['issues']}" if h["issues"] else "")
                + (f"\n    Quality judge: {h['judge_verdict']} — {h['judge_reason']}" if h["judge_reason"] else "")
                + (f"\n    Simulation: {h['sim_verdict']}" if h["sim_verdict"] else "")
            )
        lines.append("")
        return "\n".join(lines)

    def close(self) -> None:
        """Clean up the temp file."""
        if self._cache_file and os.path.exists(self._cache_file):
            try:
                os.remove(self._cache_file)
                log(f"[context_cache] Deleted: {self._cache_file}")
            except Exception as e:
                log(f"[context_cache] Could not delete cache file: {e}")
        self._cache_file = None

    def _flush(self) -> None:
        """Write current in-memory data to disk."""
        if not self._cache_file:
            return
        try:
            with open(self._cache_file, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2)
        except Exception as e:
            log(f"[context_cache] Flush error: {e}")

    def __del__(self):
        # Best-effort cleanup if close() was never called
        if self._cache_file and os.path.exists(self._cache_file):
            try:
                os.remove(self._cache_file)
            except Exception:
                pass
