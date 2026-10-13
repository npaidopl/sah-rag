"""Token-window generation for child chunking."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TokenWindow:
    """One half-open token window."""

    start: int
    end: int

    def __post_init__(self) -> None:
        """Validate the token window."""

        if self.start < 0:
            raise ValueError(
                "Token-window start cannot be negative."
            )

        if self.end <= self.start:
            raise ValueError(
                "Token-window end must exceed start."
            )

    @property
    def length(self) -> int:
        """Return the number of tokens."""

        return self.end - self.start


def build_token_windows(
    *,
    token_count: int,
    chunk_size: int,
    chunk_overlap: int,
    minimum_final_chunk_tokens: int,
) -> tuple[TokenWindow, ...]:
    """Build overlapping windows without tiny final tails."""

    if token_count < 0:
        raise ValueError(
            "token_count cannot be negative."
        )

    if token_count == 0:
        return ()

    if chunk_size < 1:
        raise ValueError(
            "chunk_size must be positive."
        )

    if chunk_overlap < 0:
        raise ValueError(
            "chunk_overlap cannot be negative."
        )

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size."
        )

    if (
        minimum_final_chunk_tokens < 1
        or minimum_final_chunk_tokens > chunk_size
    ):
        raise ValueError(
            "minimum_final_chunk_tokens must be between "
            "1 and chunk_size."
        )

    if token_count <= chunk_size:
        return (
            TokenWindow(
                start=0,
                end=token_count,
            ),
        )

    step_size = chunk_size - chunk_overlap

    windows: list[TokenWindow] = []
    start = 0

    while start < token_count:
        end = min(
            start + chunk_size,
            token_count,
        )

        #remaining = token_count - end

        if (
            end == token_count
            and end - start
            < minimum_final_chunk_tokens
            and windows
        ):
            anchored_start = max(
                0,
                token_count - chunk_size,
            )

            anchored_window = TokenWindow(
                start=anchored_start,
                end=token_count,
            )

            if anchored_window != windows[-1]:
                windows.append(anchored_window)

            break

        windows.append(
            TokenWindow(
                start=start,
                end=end,
            )
        )

        if end == token_count:
            break

        next_start = start + step_size
        next_remaining = token_count - next_start

        if (
            next_remaining
            < minimum_final_chunk_tokens
        ):
            next_start = max(
                0,
                token_count - chunk_size,
            )

        if next_start <= start:
            break

        start = next_start

    deduplicated: list[TokenWindow] = []

    for window in windows:
        if (
            not deduplicated
            or window != deduplicated[-1]
        ):
            deduplicated.append(window)

    return tuple(deduplicated)
