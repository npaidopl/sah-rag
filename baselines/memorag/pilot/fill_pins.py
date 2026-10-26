"""Fill <PIN_ME> fields in MemoRAG pilot manifests.

Deterministic fields are derived from the environment (timestamp, HF
revisions, local cache, VRAM, memorag install commit). Judgment fields
(generator identity, clue count) are left as <PIN_ME> with explicit
instructions unless overridden via CLI.

Run on the GPU host after installing memorag. On a laptop, run with
--allow-partial to see what can be filled; the smoke test will still
refuse to run until all critical fields are resolved.

Usage:
    python fill_pins.py --manifest native_manifest.yaml
    python fill_pins.py --manifest native_manifest.yaml --write
    python fill_pins.py --manifest native_manifest.yaml --write --generator-enabled true
    python fill_pins.py --manifest controlled_manifest.yaml --write --allow-partial
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError:
        sys.exit("PyYAML missing. pip install pyyaml")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def find_pins(obj: Any, path: str = "") -> list[tuple[str, str]]:
    """Return (dotted_path, current_value) for every <PIN_ME> string."""
    hits: list[tuple[str, str]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}.{k}" if path else k
            hits.extend(find_pins(v, p))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            hits.extend(find_pins(v, f"{path}[{i}]"))
    elif isinstance(obj, str) and "<PIN_ME" in obj:
        hits.append((path, obj))
    return hits


def set_dotted(obj: Any, dotted: str, value: Any) -> None:
    """Set a nested dict value by dotted path. Lists not supported."""
    if "[" in dotted:
        raise ValueError(f"list path not supported in set_dotted: {dotted}")
    parts = dotted.split(".")
    cur = obj
    for part in parts[:-1]:
        cur = cur[part]
    cur[parts[-1]] = value


def try_hf_revision(model_id: str) -> Optional[str]:
    try:
        from huggingface_hub import HfApi  # type: ignore
    except ImportError:
        return None
    try:
        return HfApi().model_info(model_id).sha
    except Exception:
        return None


def try_local_hf_revision(model_id: str) -> Optional[str]:
    cache_name = "models--" + model_id.replace("/", "--")
    candidates = [
        Path("D:/rag_work/hf_cache"),
        Path.home() / ".cache" / "huggingface" / "hub",
    ]
    for base in candidates:
        refs = base / cache_name / "refs" / "main"
        if refs.exists():
            return refs.read_text(encoding="utf-8").strip()
    return None


def try_memorag_commit() -> Optional[str]:
    try:
        from importlib.metadata import distribution  # type: ignore
        dist = distribution("memorag")
    except Exception:
        return None
    try:
        text = dist.read_text("direct_url.json")
        if not text:
            return None
        url = json.loads(text).get("url", "")
        if "@" in url:
            return url.rsplit("@", 1)[1].split("#", 1)[0]
    except Exception:
        return None
    return None


def try_cuda() -> tuple[Optional[float], Optional[str]]:
    try:
        import torch  # type: ignore
        if not torch.cuda.is_available():
            return None, None
        p = torch.cuda.get_device_properties(0)
        return round(p.total_memory / (1024 ** 3), 2), p.name
    except Exception:
        return None, None


def suggest_max_tokens(vram_gb: Optional[float]) -> int:
    if vram_gb is None:
        return 32000
    if vram_gb < 20:
        return 8000
    if vram_gb < 40:
        return 32000
    if vram_gb < 80:
        return 100000
    return 400000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--generator-enabled", choices=["true", "false"], default=None)
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--min-vram-gb", type=float, default=None)
    ap.add_argument("--allow-partial", action="store_true")
    args = ap.parse_args()

    if not args.manifest.exists():
        print(f"Manifest not found: {args.manifest}", file=sys.stderr)
        return 2

    manifest = load_manifest(args.manifest)
    original = find_pins(manifest)
    print(f"Manifest: {args.manifest}")
    print(f"Found {len(original)} <PIN_ME> field(s):")
    for p, v in original:
        print(f"  - {p}: {v}")
    print()

    filled: list[tuple[str, Any]] = []
    skipped: list[tuple[str, str]] = []

    def fill(p: str, v: Any, why: str) -> None:
        set_dotted(manifest, p, v)
        filled.append((p, v))
        print(f"  [FILL] {p} = {v!r}  ({why})")

    def skip(p: str, why: str) -> None:
        skipped.append((p, why))
        print(f"  [KEEP] {p}  ({why})")

    print("Attempting deterministic fill:")
    now_iso = datetime.now(timezone.utc).isoformat()

    for pin_path, _ in original:
        if pin_path.endswith("frozen_at_utc"):
            fill(pin_path, now_iso, "current UTC")
            continue

        if pin_path.endswith("source.commit"):
            c = try_memorag_commit()
            if c:
                fill(pin_path, c, "from importlib.metadata direct_url.json")
            else:
                skip(pin_path, "run `git rev-parse HEAD` in memorag clone")
            continue

        if pin_path.endswith("models.memory.revision"):
            try:
                mid = manifest["models"]["memory"]["id"]
            except KeyError:
                skip(pin_path, "models.memory.id missing")
                continue
            rev = try_hf_revision(mid) or try_local_hf_revision(mid)
            if rev:
                fill(pin_path, rev, f"HF revision for {mid}")
            else:
                skip(pin_path, f"cannot resolve revision for {mid}")
            continue

        if pin_path.endswith("models.retriever.revision"):
            try:
                mid = manifest["models"]["retriever"]["id"]
            except KeyError:
                skip(pin_path, "models.retriever.id missing")
                continue
            rev = try_local_hf_revision(mid) or try_hf_revision(mid)
            if rev:
                fill(pin_path, rev, f"HF revision for {mid}")
            else:
                skip(pin_path, f"cannot resolve revision for {mid}")
            continue

        if pin_path.endswith("models.injected_embedder.revision"):
            try:
                mid = manifest["models"]["injected_embedder"]["id"]
            except KeyError:
                skip(pin_path, "models.injected_embedder.id missing")
                continue
            rev = try_local_hf_revision(mid) or try_hf_revision(mid)
            if rev:
                fill(pin_path, rev, f"HF revision for {mid}")
            else:
                skip(pin_path, f"cannot resolve revision for {mid}")
            continue

        if pin_path.endswith("models.generator.enabled"):
            if args.generator_enabled is not None:
                fill(pin_path, args.generator_enabled == "true", "CLI")
            else:
                skip(pin_path, "use --generator-enabled true|false")
            continue

        if ".models.generator." in pin_path or pin_path.startswith("models.generator."):
            gen = manifest.get("models", {}).get("generator", {})
            if gen.get("enabled") is False:
                fill(pin_path, None, "generation disabled; not used")
            else:
                skip(pin_path, "set generator id/revision, or use --generator-enabled false")
            continue

        if ".models.injected_llm." in pin_path or pin_path.startswith("models.injected_llm."):
            fill(pin_path, "not_used_in_pilot", "orchestrator injects at real run")
            continue

        if pin_path.endswith("context.max_tokens"):
            if args.max_tokens is not None:
                fill(pin_path, args.max_tokens, "CLI")
            else:
                vram_gb, dev = try_cuda()
                s = suggest_max_tokens(vram_gb)
                note = f"suggested for {vram_gb}GB VRAM ({dev})" if vram_gb else "no CUDA; conservative default"
                fill(pin_path, s, note + "; override with --max-tokens")
            continue

        if pin_path.endswith("context.clue_count"):
            skip(pin_path, "check memorag package default or paper; set manually")
            continue

        if pin_path.endswith("execution.min_vram_gb"):
            if args.min_vram_gb is not None:
                fill(pin_path, args.min_vram_gb, "CLI")
            else:
                vram_gb, _ = try_cuda()
                if vram_gb:
                    fill(pin_path, round(vram_gb * 0.85, 1), f"85% of {vram_gb}GB")
                else:
                    skip(pin_path, "no CUDA; set manually or run on GPU host")
            continue

        skip(pin_path, "no auto-fill rule")

    print()
    print(f"Filled: {len(filled)} | Skipped: {len(skipped)}")
    remaining = find_pins(manifest)
    print(f"Remaining <PIN_ME>: {len(remaining)}")
    for p, v in remaining:
        print(f"  - {p}: {v}")

    if remaining and args.write and not args.allow_partial:
        print()
        print("Refusing --write while <PIN_ME> fields remain.", file=sys.stderr)
        print("Re-run with --allow-partial if this is intentional.", file=sys.stderr)
        return 1

    if args.write:
        try:
            import yaml  # type: ignore
        except ImportError:
            sys.exit("PyYAML missing")
        args.manifest.write_text(
            yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        print(f"\nWROTE: {args.manifest}")
    else:
        print("\nDRY RUN. Re-run with --write to persist.")
    return 0


if __name__ == "__main__":
    sys.exit(main())