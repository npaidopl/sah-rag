#!/usr/bin/env python3
"""Phase 2: rewrite `rag_system.*` imports to v2 layout.

Ordered replacements: specific -> general. Uses word boundaries so
`rag_system.evaluation_contracts` is not partially matched by
`rag_system.evaluation`.

Usage:
    python scripts/phase2_rewrite.py            # dry-run
    python scripts/phase2_rewrite.py --apply    # write changes
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APPLY = "--apply" in sys.argv

RULES: list[tuple[str, str]] = [
    # --- "from rag_system import X" for moved modules ---
    (r"from rag_system import baselines\b",              "import baselines"),
    (r"from rag_system import evaluation\b",             "import evaluation"),
    (r"from rag_system import orchestration\b",          "import orchestration"),
    (r"from rag_system import evaluation_contracts\b",   "from evaluation import contracts"),
    (r"from rag_system import evaluation_persistence\b", "from evaluation import persistence"),
    (r"from rag_system import execution_contracts\b",    "from orchestration import contracts"),
    (r"from rag_system import orchestration_persistence\b", "from orchestration import persistence"),
    (r"from rag_system import pilot_dataset\b",          "from datasets import pilot_synthetic"),
    (r"from rag_system import pilot_dataset_persistence\b", "from datasets import pilot_persistence"),

    # --- submodule imports for moved modules (specific first) ---
    (r"\brag_system\.pilot_dataset_persistence\b",   "datasets.pilot_persistence"),
    (r"\brag_system\.pilot_dataset\b",               "datasets.pilot_synthetic"),
    (r"\brag_system\.evaluation_contracts\b",        "evaluation.contracts"),
    (r"\brag_system\.evaluation_persistence\b",      "evaluation.persistence"),
    (r"\brag_system\.evaluation\b",                  "evaluation"),
    (r"\brag_system\.execution_contracts\b",         "orchestration.contracts"),
    (r"\brag_system\.orchestration_persistence\b",   "orchestration.persistence"),
    (r"\brag_system\.orchestration\b",               "orchestration"),
    (r"\brag_system\.baselines\b",                   "baselines"),

    # --- catch-all: whatever remained goes under sah_rag ---
    (r"\brag_system\.",  "sah_rag."),
    (r"\brag_system\b",  "sah_rag"),
]

TARGET_DIRS = [
    "sah_rag", "baselines", "evaluation", "orchestration",
    "datasets", "scripts", "tests",
]
EXCLUDE_FILES = {"scripts/phase2_rewrite.py"}


def iter_files() -> list[Path]:
    files: list[Path] = []
    for d in TARGET_DIRS:
        p = ROOT / d
        if not p.exists():
            continue
        for f in p.rglob("*.py"):
            rel = f.relative_to(ROOT).as_posix()
            if rel in EXCLUDE_FILES:
                continue
            files.append(f)
    return files


def rewrite(text: str) -> tuple[str, list[tuple[str, int]]]:
    changes: list[tuple[str, int]] = []
    for pattern, replacement in RULES:
        new_text, n = re.subn(pattern, replacement, text)
        if n:
            changes.append((pattern, n))
            text = new_text
    return text, changes


def main() -> int:
    files = iter_files()
    print(f"Mode: {'APPLY' if APPLY else 'DRY-RUN'}")
    print(f"Scanning {len(files)} .py files in {TARGET_DIRS}")
    print()

    modified: list[tuple[Path, list[tuple[str, int]]]] = []
    by_rule: Counter = Counter()

    for f in files:
        try:
            original = f.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            print(f"SKIP (encoding): {f.relative_to(ROOT)}")
            continue
        new_text, changes = rewrite(original)
        if not changes:
            continue
        modified.append((f, changes))
        for pattern, n in changes:
            by_rule[pattern] += n
        if APPLY:
            f.write_text(new_text, encoding="utf-8")

    total = sum(by_rule.values())
    print(f"Files to modify: {len(modified)}")
    print(f"Total replacements: {total}")
    print()

    if by_rule:
        print("=== Replacements per rule ===")
        for pattern, n in by_rule.most_common():
            print(f"  {n:5d}  {pattern}")
        print()

    print("=== First 25 modified files ===")
    for f, _ in modified[:25]:
        print(f"  {f.relative_to(ROOT)}")

    if not APPLY and modified:
        print()
        print("Run again with --apply to write changes.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
