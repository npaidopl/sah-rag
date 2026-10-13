"""Validate the Step 93 component substitution policy."""
from __future__ import annotations

from pathlib import Path

import yaml

PATH = Path("baseline_audit/component_substitution_policy.yaml")


def _require(c: bool, m: str) -> None:
    if not c:
        raise SystemExit(m)


def main() -> int:
    d = yaml.safe_load(PATH.read_text(encoding="utf-8"))
    _require(d.get("artifact_type") == "component_substitution_policy", "artifact_type changed")

    roles = d["component_roles"]
    for r in ("ARCHITECTURE_DEFINING","ARCHITECTURE_SUPPORTING","OPTIONAL_DOWNSTREAM"):
        _require(r in roles, f"role {r} missing")

    comps = d["components"]
    for c in ("embedder","tokenizer","storage_backend","llm_index_time",
              "llm_query_time","llm_final_generation","chunker","hierarchy_builder",
              "graph_builder","reranker","persistence_format","evaluator"):
        _require(c in comps, f"component {c} missing")
        _require("default_role" in comps[c], f"{c}: default_role missing")

    st = d["storage_substitution_policy"]
    _require(
        isinstance(st["common_qdrant_allowed_when"], list),
        "common_qdrant_allowed_when must be list",
    )
    _require(
        isinstance(st["native_storage_required_when"], list),
        "native_storage_required_when must be list",
    )

    llm = d["llm_policy"]
    _require(llm["final_generation"]["enabled"] is False, "final_generation must be disabled")
    _require(llm["index_time"]["preserve_when_architecture_defining"] is True,
             "index_time preserve_when_architecture_defining must be true")
    _require(llm["query_time"]["preserve_when_architecture_defining"] is True,
             "query_time preserve_when_architecture_defining must be true")

    print("[OK] Component substitution policy")
    print("  components:", len(comps))
    print("  llm execution tracks:", len(llm["execution_tracks"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())