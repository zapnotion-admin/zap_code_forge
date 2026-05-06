from pathlib import Path

APP_DIR      = Path(__file__).parent.parent.resolve()
SESSIONS_DIR = APP_DIR / "sessions"
LOGS_DIR     = APP_DIR / "logs"
SESSIONS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

OLLAMA_URL = "http://localhost:11434"

# Models
MODEL_CODER    = "qwen3-coder"        # Qwen3 14B — writes code
MODEL_REASONER = "deepseek-reasoner"  # DeepSeek-R1 8B — plans, reasons
MODEL_FALLBACK = "qwen3:14b"          # Used if custom model hasn't been created yet

# VRAM / context limits
# RTX 4080 16GB with one model at a time.
# 24576 matches the Modelfile num_ctx — must be identical or Ollama ignores it.
# Keeping MAX_CTX slightly below Modelfile limit leaves room for the response.
MAX_CTX_CODER    = 22_000   # Qwen3 14B   (Modelfile: 24576)
MAX_CTX_REASONER = 22_000   # DeepSeek 8B (Modelfile: 24576)
MAX_TOKENS_OUTPUT = 8_192   # Max tokens the model can output per call.
                             # Ollama default is ~1024 which truncates large files.
                             # 8192 safely covers files up to ~600 lines.
MAX_FILE_CHARS   = 30_000   # Raised from 20K — more file context per prompt
MAX_FILES        = 5        # Max files user can add at once
MAX_PROMPT_CHARS = 55_000   # Raised from 40K to match larger context window

# Execution mode
# Stable: 2 retries per step (slower but more robust)
# Fast:   1 retry per step  (quicker, less overhead)
# Both modes use identical verification logic — the heuristic was removed.
STABLE_MODE = True   # Default to Stable for real projects

# Outer convergence loop
# How many full pipeline passes to allow. 1 = original single-pass behaviour.
# After each pass a quality judge decides whether to stop or try again.
# Range: 1-5. Recommended: 2 for most tasks, 3 for complex ones.
MAX_OUTER_ITERATIONS = 2   # Default: 2 passes (generate + correct). Range 1-5.

# Temperature controls
# Pipeline stages (Planner, Coder, Reviewer, Retry) use low temp for determinism.
# Interactive chat uses a higher temp for more natural responses.
TEMPERATURE_CODING   = 0.25   # Planner, Coder, Reviewer, Retry — deterministic
TEMPERATURE_PLANNING = 0.35   # Slightly more creative for planning/reasoning
TEMPERATURE_CHAT     = 0.60   # Chat/interactive streaming

# VRAM management
# Target VRAM usage in GB. Models are unloaded between outer loop iterations
# to keep usage below this threshold on the 16GB RTX 4080.
VRAM_TARGET_GB = 13.0

# Context cache
# When True, large context blobs are written to a temp file on disk between
# outer loop iterations instead of staying in memory. Trades a small amount
# of disk I/O for lower peak RAM/VRAM pressure on large projects.
CONTEXT_CACHE_ENABLED = True

# CPU thread cap for Ollama inference [perf: prevents starving the rest of the OS]
OLLAMA_NUM_THREAD = 4

# UI Palette — dark theme
PALETTE = {
    "bg":       "#0e0f11",
    "bg2":      "#14161a",
    "bg3":      "#1c1f25",
    "border":   "#2a2d35",
    "accent":   "#4ade80",   # green  — AI responses, OK states
    "accent2":  "#22d3ee",   # cyan   — user messages, links
    "warn":     "#fb923c",   # orange — thinking, warnings
    "err":      "#f87171",   # red    — errors
    "text":     "#e2e8f0",
    "text_dim": "#64748b",
    "user_msg": "#1e293b",
    "ai_msg":   "#0f1a12",
}

# VRAM estimates shown in the model selector
VRAM_ESTIMATES = {
    "qwen3-coder":       "~10 GB",
    "deepseek-reasoner": "~5.7 GB",
    "qwen3:14b":         "~10 GB",
}

# Keywords that trigger the Planner → Coder → Reviewer pipeline
TASK_KEYWORDS = [
    "add", "fix", "refactor", "implement", "create", "update",
    "change", "build", "write", "modify", "delete", "rename",
    "move", "extract", "replace", "debug", "optimise", "optimize",
]

# Python import validation — used by sanity checker for .py files
PYTHON_IMPORT_CHECK = True   # attempt `python3 -m py_compile` on generated .py files
