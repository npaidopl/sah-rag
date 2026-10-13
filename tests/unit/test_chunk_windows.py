"""Tests for token-window generation."""

from __future__ import annotations

from rag_system.chunking import build_token_windows


def test_short_input_creates_one_window() -> None:
    """Input shorter than chunk size needs one window."""

    windows = build_token_windows(
        token_count=6,
        chunk_size=8,
        chunk_overlap=4,
        minimum_final_chunk_tokens=4,
    )

    assert [
        (window.start, window.end)
        for window in windows
    ] == [(0, 6)]


def test_regular_windows_preserve_overlap() -> None:
    """Normal windows should use the configured overlap."""

    windows = build_token_windows(
        token_count=20,
        chunk_size=8,
        chunk_overlap=4,
        minimum_final_chunk_tokens=4,
    )

    assert [
        (window.start, window.end)
        for window in windows
    ] == [
        (0, 8),
        (4, 12),
        (8, 16),
        (12, 20),
    ]


def test_tiny_tail_is_anchored_to_end() -> None:
    """A tiny final tail should become a full anchored window."""

    windows = build_token_windows(
        token_count=15,
        chunk_size=8,
        chunk_overlap=4,
        minimum_final_chunk_tokens=4,
    )

    assert [
        (window.start, window.end)
        for window in windows
    ] == [
        (0, 8),
        (4, 12),
        (8, 15),
    ]


def test_every_token_is_covered() -> None:
    """Every source token must occur in at least one window."""

    token_count = 31

    windows = build_token_windows(
        token_count=token_count,
        chunk_size=8,
        chunk_overlap=4,
        minimum_final_chunk_tokens=4,
    )

    covered = set()

    for window in windows:
        covered.update(
            range(window.start, window.end)
        )

    assert covered == set(range(token_count))


def test_empty_input_creates_no_windows() -> None:
    """Zero tokens require no child chunks."""

    assert build_token_windows(
        token_count=0,
        chunk_size=8,
        chunk_overlap=4,
        minimum_final_chunk_tokens=4,
    ) == ()
