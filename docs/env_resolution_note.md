# Environment Resolution Note

**Date:** 2026-11-04 (FIX-07)

## Two ways `RAG_CANONICAL_ROOT` gets resolved

### 1. Via pytest (automatic)

`tests/conftest.py` runs `_set_test_env_defaults()` at import time:

    defaults = {
        "RAG_CANONICAL_ROOT": str(PROJECT_ROOT / "data" / "processed" / "canonical"),
        "RAG_INDOOR_LLM_ROOT": "D:/rag_work/models/Qwen2.5-14B-Instruct",
    }
    for key, value in defaults.items():
        if not os.getenv(key):
            os.environ[key] = value

This means **pytest always works**, even if `.env` is missing or wrong.

### 2. Via standalone scripts (explicit)

`sah_rag.canonical.paths.load_project_environment()` loads `.env` from `get_project_root() / ".env"`.

However, `python-dotenv`'s `load_dotenv()` with no args uses `find_dotenv()`, which searches from the **caller's `__file__` directory upward** — NOT from CWD. So if you run a script from outside the repo (e.g., `$TEMP\test.py`), it will not find the repo's `.env`.

**Correct standalone pattern:**

    from pathlib import Path
    from dotenv import load_dotenv

    REPO = Path(r"C:\Users\USER1\Documents\SAH-RAG")
    load_dotenv(dotenv_path=REPO / ".env", override=False)

    from sah_rag.canonical.paths import get_canonical_root
    root = get_canonical_root()
    print(root)  # C:\Users\USER1\Documents\SAH-RAG\data\processed\canonical

## Verified

- pytest: works (conftest defaults)
- standalone (correct pattern): works
- standalone (naive `load_dotenv()` from `$TEMP`): fails, expected

This is **not a bug** — it's standard `python-dotenv` behavior.