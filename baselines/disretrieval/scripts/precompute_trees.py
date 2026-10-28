"""Precompute DISRetrieval trees via the official RSTProcesser."""
from __future__ import annotations

import argparse
import configparser
import json
import os
import sys
from pathlib import Path
from typing import Any

# -------- Monkey-patch open() to default to UTF-8 --------
import builtins
_orig_open = builtins.open
def _utf8_open(*args, **kwargs):
    mode = kwargs.get("mode", args[1] if len(args) > 1 else "r")
    if "b" not in str(mode) and "encoding" not in kwargs:
        kwargs["encoding"] = "utf-8"
    return _orig_open(*args, **kwargs)
builtins.open = _utf8_open
# ---------------------------------------------------------
DISRETRIEVAL_ROOT = Path(r"D:\rag_work\parsers\DISRetrieval_official")
UPSTREAM_MODEL = Path(r"D:\rag_work\models\upstream_enNrstparser\extracted_2\rst_model")
HF_CACHE = Path(r"D:\rag_work\hf_cache")
BGE_M3 = HF_CACHE / "models--BAAI--bge-m3"
QWEN25 = HF_CACHE / "models--Qwen--Qwen2.5-1.5B-Instruct"
QWEN2 = HF_CACHE / "models--Qwen--Qwen2-1.5B-Instruct"
OUTPUT_ROOT = Path(r"D:\rag_work\disretrieval_trees")

DATASET_SOURCES = {
    "qasper": DISRETRIEVAL_ROOT / "data" / "qasper" / "doc_data",
    "quality": DISRETRIEVAL_ROOT / "data" / "quality" / "doc_data",
    "narrative": DISRETRIEVAL_ROOT / "data" / "narrative" / "doc_data",
    "multiFieldQA-zh": DISRETRIEVAL_ROOT / "data" / "multiFieldQA-zh" / "doc_data",
}


def _snapshot(model_dir: Path) -> Path:
    snaps = model_dir / "snapshots"
    if snaps.exists():
        cand = list(snaps.iterdir())
        if cand:
            return cand[0]
    return model_dir


def _patch_torch_load_cpu() -> None:
    import torch
    orig = torch.load
    def _cpu(*a, **kw):
        kw.setdefault("map_location", "cpu")
        kw.setdefault("weights_only", False)
        return orig(*a, **kw)
    torch.load = _cpu


def _build_parser():
    _patch_torch_load_cpu()
    sys.path.insert(0, str(DISRETRIEVAL_ROOT))
    sys.path.insert(0, str(DISRETRIEVAL_ROOT / "rst_parser"))
    from rst_parser.driver.RSTParser import RSTParser

    cfg = configparser.ConfigParser()
    cfg.read(UPSTREAM_MODEL / "config.cfg")
    cfg["Data"]["xlnet_dir"] = str(UPSTREAM_MODEL / "xlm_rst_tuned")
    cfg["Save"]["save_dir"] = str(UPSTREAM_MODEL)
    cfg["Save"]["xlnet_save_dir"] = str(UPSTREAM_MODEL / "xlm_rst_tuned")
    cfg["Save"]["config_file"] = str(UPSTREAM_MODEL / "config_local.cfg")
    cfg["Save"]["save_model_path"] = str(UPSTREAM_MODEL / "model")
    cfg["Save"]["save_vocab_path"] = str(UPSTREAM_MODEL / "vocab")
    cfg["Save"]["load_dir"] = str(UPSTREAM_MODEL)
    cfg["Save"]["load_model_path"] = str(UPSTREAM_MODEL / "model")
    cfg["Save"]["load_vocab_path"] = str(UPSTREAM_MODEL / "vocab")
    local_cfg = UPSTREAM_MODEL / "config_local.cfg"
    with open(local_cfg, "w", encoding="utf-8") as f:
        cfg.write(f)

    os.chdir(DISRETRIEVAL_ROOT)
    parser = RSTParser(config_file=str(local_cfg), use_cuda=False)

    def parse(subset: dict, tmp_dir: Path) -> Path:
        tmp_in = tmp_dir / "subset.json"
        tmp_in.write_text(json.dumps(subset, ensure_ascii=False), encoding="utf-8")
        return Path(parser.start_parsing(edu_file=str(tmp_in)))

    return parse


def _build_official_processor():
    sys.path.insert(0, str(DISRETRIEVAL_ROOT / "rst_tree_build"))
    sys.path.insert(0, str(DISRETRIEVAL_ROOT / "rst_parser"))
    from RSTProcesser import RSTProcessor
    return RSTProcessor()


def _build_embedder():
    import torch
    from transformers import AutoModel, AutoTokenizer
    path = str(_snapshot(BGE_M3))
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModel.from_pretrained(path, local_files_only=True).to("cpu")
    model.eval()
    torch.set_grad_enabled(False)

    def encode(texts: list[str], batch_size: int = 16) -> list[list[float]]:
        out = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
            with torch.no_grad():
                result = model(**inputs)
            dense = torch.nn.functional.normalize(result.last_hidden_state[:, 0], p=2, dim=1)
            out.extend(dense.cpu().tolist())
        return out

    return encode


def _convert_official_tree(node: dict, depth: int = 0) -> dict[str, Any]:
    if node["is_leaf"]:
        text = node.get("node_text") or ""
        return {
            "id": node["node_id"],
            "text": text,
            "summary": "",
            "children": [],
            "level": depth,
            "is_leaf": True,
            "node_interval": node.get("node_interval"),
        }
    left = _convert_official_tree(node["left_child"], depth + 1)
    right = _convert_official_tree(node["right_child"], depth + 1)
    text = " ".join(c["text"] for c in (left, right) if c.get("text")).strip()
    return {
        "id": node["node_id"],
        "text": text,
        "summary": "",
        "children": [left, right],
        "level": depth,
        "is_leaf": False,
        "node_interval": node.get("node_interval"),
    }


def _collect_nodes(tree: dict) -> list[dict]:
    nodes = [tree]
    for c in tree.get("children", []):
        nodes.extend(_collect_nodes(c))
    return nodes


def _fill_summaries(tree: dict, llm_summarize=None) -> dict:
    children = tree.get("children") or []
    if not children:
        text = tree.get("text") or ""
        tree["summary"] = text
        return tree
    new_children = [_fill_summaries(c, llm_summarize) for c in children]
    parts = [(c.get("text") or "") for c in new_children]
    joined = " ".join(parts).strip()
    if llm_summarize:
        try:
            summary = llm_summarize(joined[:1500], max_tokens=100) or ""
        except Exception as e:
            print(f"  summarizer failed: {e}")
            summary = joined[:200]
    else:
        summary = joined[:300]
    if not summary:
        summary = joined[:300] or (tree.get("text") or "")
    return {**tree, "children": new_children, "summary": summary}


def _build_summarizer():
    """Return (summarize_callable, used_model_name)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    candidates = [QWEN25, QWEN2]
    last_err = None
    for cand in candidates:
        if not cand.exists():
            continue
        try:
            path = str(_snapshot(cand))
            print(f"Trying summarizer at: {path}")
            tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(
                path, local_files_only=True
            ).to("cpu")
            model.eval()
            torch.set_grad_enabled(False)

            def summarize(context: str, max_tokens: int = 100) -> str:
                prompt = (
                    "Summarize the following passage in one concise sentence. "
                    "Preserve entities, numbers, and factual claims. "
                    "Do not add information not present in the passage.\n\n"
                    f"{context[:1500]}\n\nSummary:"
                )
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=2048,
                )
                with torch.no_grad():
                    out = model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                gen = out[0][inputs["input_ids"].shape[1]:]
                text_out = tokenizer.decode(gen, skip_special_tokens=True).strip()
                return text_out or "(empty)"

            print(f"Summarizer loaded: {cand.name}")
            return summarize, cand.name
        except Exception as e:
            last_err = e
            print(f"Failed to load {cand.name}: {e}")
            continue
    raise RuntimeError(f"No usable summarizer. Last error: {last_err}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None,
                    help="Dataset name (one of the built-in list) or use --data-dir")
    ap.add_argument("--data-dir", default=None,
                    help="Explicit path to a doc_data directory containing subtree.json + subtree_sentmap.json")
    ap.add_argument("--mode", choices=["native", "controlled"], default="controlled",
                    help="Component mode (default: controlled)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip-summarization", action="store_true")
    ap.add_argument("--skip-embeddings", action="store_true")
    args = ap.parse_args()

    if args.data_dir:
        src = Path(args.data_dir)
        if not src.exists():
            print(f"ERROR: --data-dir not found: {src}", file=sys.stderr)
            return 1
        dataset_name = src.parent.name if src.name == "doc_data" else src.name
    elif args.dataset:
        if args.dataset not in DATASET_SOURCES:
            print(f"ERROR: --dataset {args.dataset!r} not in built-in list. "
                  f"Use --data-dir for custom paths.", file=sys.stderr)
            return 1
        src = DATASET_SOURCES[args.dataset]
        dataset_name = args.dataset
    else:
        print("ERROR: must specify --dataset or --data-dir", file=sys.stderr)
        return 1

    # Native mode requires paper's backbone
    if args.mode == "native":
        if not NATIVE_BACKBONE.exists():
            print(f"ERROR: native mode requires paper backbone at {NATIVE_BACKBONE}\n"
                  f"       Missing. Use --mode controlled instead.", file=sys.stderr)
            return 1
    subtree_file = src / "subtree.json"
    sentmap_file = src / "subtree_sentmap.json"
    if not subtree_file.exists() or not sentmap_file.exists():
        print(f"ERROR: missing {subtree_file} or {sentmap_file}", file=sys.stderr)
        return 1

    all_docs = json.loads(subtree_file.read_text(encoding="utf-8"))
    if args.limit:
        all_docs = dict(list(all_docs.items())[: args.limit])
    print(f"Dataset: {dataset_name} (mode={args.mode}), documents: {len(all_docs)}")

    out_dir = OUTPUT_ROOT / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir / "_tmp"
    import shutil as _shutil
    if tmp_dir.exists():
        _shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    print("Loading parser...")
    parse = _build_parser()
    print("Loading RSTProcessor...")
    processor = _build_official_processor()

    print(f"Parsing {len(all_docs)} docs...")
    rst_path = parse(all_docs, tmp_dir)
    if rst_path.exists():
        print(f"RST output size: {rst_path.stat().st_size} bytes")
        if rst_path.stat().st_size < 100:
            print(f"WARN: tiny output, preview: {rst_path.read_text(encoding='utf-8', errors='replace')[:300]}")
    print(f"RST output: {rst_path}")

    official_dir = out_dir / "_official_trees"
    official_dir.mkdir(exist_ok=True)
    processor.process_rst_and_construct_trees(
        rst_file=str(rst_path),
        paragraph_edu_file=str(subtree_file),
        tree_path=str(official_dir),
        sent_map_path=str(sentmap_file),
    )
    print(f"Official trees built: {len(list(official_dir.glob('*.json')))} files")

    embed = None
    if not args.skip_embeddings:
        print("Loading BGE-M3...")
        embed = _build_embedder()

    llm_summarize = None
    if not args.skip_summarization:
        print("Loading Qwen2.5-1.5B-Instruct summarizer...")
        llm_summarize, _used_summarizer = _build_summarizer()

    for i, doc_id in enumerate(all_docs.keys(), 1):
        official_tree_file = official_dir / f"{doc_id}.json"
        if not official_tree_file.exists():
            print(f"[{i}/{len(all_docs)}] {doc_id}: MISSING (official tree)")
            continue
        official = json.loads(official_tree_file.read_text(encoding="utf-8"))
        tree = _convert_official_tree(official["full_tree"])
        tree["doc_id"] = doc_id
        tree = _fill_summaries(tree, llm_summarize)

        embeddings = {}
        embedder_meta = {}
        if embed is not None:
            nodes = _collect_nodes(tree)
            texts = [n.get("summary") or n.get("text", "") for n in nodes]
            vecs = embed(texts)
            for n, v in zip(nodes, vecs):
                embeddings[n["id"]] = v
            embedder_meta = {"name": "bge_m3", "dim": len(vecs[0]) if vecs else 0}

        payload = {
            "doc_id": doc_id,
            "tree": tree,
            "embeddings": embeddings,
            "embedder": embedder_meta,
        }
        (out_dir / f"doc_{doc_id}.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )
        leaf_count = sum(1 for n in _collect_nodes(tree) if not n["children"])
        print(f"[{i}/{len(all_docs)}] {doc_id}: {leaf_count} leaves, {len(embeddings)} embeddings")

    manifest = {
        "dataset": dataset_name,
        "document_count": len(all_docs),
        "parser": "E-NNRSTParser upstream (Apache-2.0)",
        "processor": "RSTProcesser (official DISRetrieval)",
        "summarizer": "concat (no LLM)" if args.skip_summarization else "Qwen (pending)",
        "embedder": "skipped" if embed is None else "bge_m3",
        "native_fidelity": "CONTROLLED_ADAPTATION",
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nDone. Output: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())