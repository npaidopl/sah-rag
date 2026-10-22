"""Pytest config for BM25 baseline tests.

Problem
-------
On Windows, `pytest`'s built-in `faulthandler` plugin emits false-positive
"Windows fatal exception: access violation" traces whenever `jnius` (used by
Pyserini to load the JVM) initializes. The JVM uses SIGSEGV internally for
safepoint polling; `faulthandler` interprets this as a crash even though the
JVM is healthy.

Why `unregister(name="faulthandler")` alone is not enough
--------------------------------------------------------
Pytest registers its `faulthandler` plugin very early in `_pytest.config`.
By the time `pytest_configure` runs (even with `trylast=True` on our side),
pytest has already called `faulthandler.enable()`, so the noise is already
happening by the time we can unregister.

Fix
---
Monkeypatch `faulthandler.enable` to a no-op at *module import time* of this
`conftest.py` (before any pytest hook runs). Also call `faulthandler.disable()`
to turn off whatever is currently active.

This removes the noise WITHOUT hiding real failures:
- If the process truly segfaults, pytest exits non-zero.
- If an assertion fails, pytest reports it normally.
The `faulthandler` module is purely diagnostic.

`pyproject.toml`, `tests/conftest.py` (project-level) and `baselines/base.py`
are NOT touched.
"""
import faulthandler as _faulthandler

# Silence whatever is currently active.
_faulthandler.disable()

# Prevent pytest (or anything else) from re-enabling it later.
_faulthandler.enable = lambda *args, **kwargs: None  # noqa: ARG005