"""Validate the architecture portfolio readiness snapshot."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT = (
    REPO_ROOT
    / "baseline_audit"
    / "architecture_portfolio_readiness_snapshot.yaml"
)


def _fail(msg: str) -> None:
    print(f"[FAIL] {msg}")
    sys.exit(1)


def _is_dataset_ready(entry: dict[str, Any]) -> bool:
    return (
        entry.get("implementation_status") == "DATASET_READY"
        and entry.get("readiness_gate") == "PASSED"
        and entry.get("dataset_readiness_status") == "READY"
    )


def _is_deferred(entry: dict[str, Any]) -> bool:
    return entry.get("controlled_track_status") == "DEFERRED"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", default=DEFAULT, type=Path)
    args = parser.parse_args()
    if not args.snapshot.exists():
        _fail(f"snapshot not found: {args.snapshot}")
    with args.snapshot.open("r", encoding="utf-8") as f:
        data: Any = yaml.safe_load(f)
    if not isinstance(data, dict):
        _fail("not a mapping")
    if data.get("artifact_type") != (
        "architecture_portfolio_readiness_snapshot"
    ):
        _fail("artifact_type mismatch")
    snap = data.get("snapshot") or {}
    if snap.get("status") != "FROZEN":
        _fail("snapshot.status must be FROZEN")
    cs = data.get("counter_source") or {}
    if cs.get("type") != "ARCHITECTURE_READINESS_RECORDS":
        _fail("counter_source.type mismatch")
    if cs.get("manual_counter_edits_allowed") is not False:
        _fail("manual_counter_edits_allowed must be false")

    architectures = data.get("architectures") or []
    if not architectures:
        _fail("architectures is empty")

    # Recompute the counters from records.
    ds_members: list[str] = []
    def_members: list[str] = []
    for a in architectures:
        aid = a.get("architecture_id")
        if not isinstance(aid, str) or not aid:
            _fail("architecture entry missing architecture_id")
        if _is_dataset_ready(a):
            ds_members.append(aid)
        if _is_deferred(a):
            def_members.append(aid)

    counters = data.get("counters") or {}
    if int(counters.get("dataset_ready_count", -1)) != len(ds_members):
        _fail(
            f"dataset_ready_count {counters.get('dataset_ready_count')} "
            f"!= actual {len(ds_members)}"
        )
    if int(counters.get("deferred_count", -1)) != len(def_members):
        _fail(
            f"deferred_count {counters.get('deferred_count')} "
            f"!= actual {len(def_members)}"
        )
    members = data.get("dataset_ready_members") or []
    if sorted(members) != sorted(ds_members):
        _fail(
            f"dataset_ready_members {members} != actual {ds_members}"
        )
    def_members_doc = data.get("deferred_members") or []
    if sorted(def_members_doc) != sorted(def_members):
        _fail(
            f"deferred_members {def_members_doc} != actual "
            f"{def_members}"
        )

    print(
        f"[OK] portfolio readiness snapshot "
        f"(dataset_ready={len(ds_members)}, "
        f"deferred={len(def_members)})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())