"""Read-only introspection of the project-owned LongRAG execution API."""
from __future__ import annotations

import importlib
import inspect
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


CANDIDATE_MODULES = [
    "rag_system.baselines.longrag",
    "rag_system.baselines.longrag_embeddings",
    "rag_system.baselines.longrag_retrieval",
    "rag_system.baselines.longrag_storage",
]


def describe_member(obj: object) -> dict:
    try:
        sig = str(inspect.signature(obj))
    except (TypeError, ValueError):
        sig = "<no signature>"
    return {
        "kind": "class" if inspect.isclass(obj) else "function",
        "signature": sig,
        "module": getattr(obj, "__module__", None),
        "qualname": getattr(obj, "__qualname__", None),
        "doc_first_line": (inspect.getdoc(obj) or "").split("\n")[0],
    }


summary: dict = {}
for module_name in CANDIDATE_MODULES:
    try:
        mod = importlib.import_module(module_name)
    except Exception as exc:
        summary[module_name] = {"import_error": f"{type(exc).__name__}: {exc}"}
        continue

    members: dict = {}
    for name, obj in inspect.getmembers(mod):
        if name.startswith("_"):
            continue
        if inspect.isfunction(obj) or inspect.isclass(obj):
            if not getattr(obj, "__module__", "").startswith("rag_system"):
                continue
            members[name] = describe_member(obj)
    summary[module_name] = members

print("===== LONGRAG EXTENDED API INVENTORY (JSON) =====")
print(json.dumps(summary, indent=2, sort_keys=True))