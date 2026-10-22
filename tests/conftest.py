"""Pytest configuration and shared fixtures for SAH-RAG v2."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _set_test_env_defaults() -> None:
    """Ensure critical env vars are set so tests don't crash on missing config."""
    defaults: dict[str, str] = {
        "RAG_CANONICAL_ROOT": str(PROJECT_ROOT / "processed" / "canonical"),
    }
    for key, value in defaults.items():
        if not os.getenv(key):
            os.environ[key] = value


_set_test_env_defaults()


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Absolute path to the project root."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def canonical_root() -> Path:
    """Absolute path to the canonical datasets root, skipped if missing."""
    path = Path(os.environ["RAG_CANONICAL_ROOT"])
    if not path.exists():
        pytest.skip(f"RAG_CANONICAL_ROOT does not exist: {path}")
    return path


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "integration: requires external service (e.g., Qdrant server)",
    )
    config.addinivalue_line("markers", "gpu: requires GPU")
    config.addinivalue_line("markers", "slow: slow-running test")
