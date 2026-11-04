"""Phase 0.1 smoke: verify config paths, 17 datasets, BGE-M3 weights."""
from ablation.config import (
    SAH_RAG_ROOT, CANONICAL_DATA_ROOT, WORK_ROOT, HF_CACHE_ROOT,
    ALL_DATASETS, TIER1_VARIANTS, TIER2_VARIANTS, TIER3_VARIANTS, TIER3_DATASETS,
    EXCLUDE_FROM_MACRO, ensure_dirs,
)
from ablation.hf_cache import resolve_snapshot

def main() -> int:
    ensure_dirs()
    print(f"SAH_RAG_ROOT:        {SAH_RAG_ROOT}")
    print(f"CANONICAL_DATA_ROOT: {CANONICAL_DATA_ROOT}")
    print(f"WORK_ROOT:           {WORK_ROOT}")
    print(f"HF_CACHE_ROOT:       {HF_CACHE_ROOT}")
    print()

    # Verify 17 datasets
    missing = []
    for ds in ALL_DATASETS:
        d = CANONICAL_DATA_ROOT / ds
        if not ((d / "documents.jsonl").is_file() and (d / "examples.jsonl").is_file()):
            missing.append(ds)
    print(f"Datasets OK: {len(ALL_DATASETS) - len(missing)}/{len(ALL_DATASETS)}")
    if missing:
        print(f"  MISSING: {missing}")
    print()

    # Tier totals
    n_macro = len(ALL_DATASETS) - len(EXCLUDE_FROM_MACRO)
    t1 = len(TIER1_VARIANTS) * len(ALL_DATASETS)
    t2 = len(TIER2_VARIANTS) * len(ALL_DATASETS)
    t3 = len(TIER3_VARIANTS) * len(TIER3_DATASETS)
    print(f"Macro datasets (excl. {sorted(EXCLUDE_FROM_MACRO)}): {n_macro}")
    print(f"Tier 1 embeds:   {t1}  ({len(TIER1_VARIANTS)} variants x 17)")
    print(f"Tier 2 re-ranks: {t2}  ({len(TIER2_VARIANTS)} variants x 17, FREE)")
    print(f"Tier 3 embeds:   {t3}  ({len(TIER3_VARIANTS)} variants x {len(TIER3_DATASETS)})")
    print(f"Total GPU jobs:  {t1 + t3}")
    print(f"Total CPU jobs:  {t1 + t2 + t3}")
    print()

    # Resolve BGE-M3
    try:
        bge = resolve_snapshot("BAAI/bge-m3")
        print(f"BGE-M3 snapshot: {bge}")
        print(f"  safetensors:        {any(bge.glob('*.safetensors'))}")
        print(f"  pytorch_model.bin:  {(bge / 'pytorch_model.bin').is_file()}")
        print(f"  sparse_linear.pt:   {(bge / 'sparse_linear.pt').is_file()}")
        print(f"  colbert_linear.pt:  {(bge / 'colbert_linear.pt').is_file()}")
    except FileNotFoundError as e:
        print(f"BGE-M3 FAIL: {e}")
        return 1

    return 0 if not missing else 2

if __name__ == "__main__":
    raise SystemExit(main())