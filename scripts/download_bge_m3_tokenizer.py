"""Download and inspect only the BGE-M3 tokenizer files."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from transformers import AutoTokenizer

MODEL_ID = "BAAI/bge-m3"
REPORT_PATH = Path(
    "outputs/bge_m3_tokenizer_download.json"
)


def directory_size(path: Path) -> int:
    """Return the recursive size of an existing directory."""

    if not path.exists():
        return 0

    return sum(
        item.stat().st_size
        for item in path.rglob("*")
        if item.is_file()
    )


def main() -> None:
    """Download tokenizer artifacts without model weights."""

    load_dotenv(
        dotenv_path=Path(".env"),
        override=False,
    )

    hf_home_raw = os.getenv("HF_HOME")

    if not hf_home_raw:
        raise RuntimeError(
            "HF_HOME is not configured in .env."
        )

    hf_home = Path(hf_home_raw)
    hf_home.mkdir(
        parents=True,
        exist_ok=True,
    )

    size_before = directory_size(hf_home)

    print("[INFO] Model ID:", MODEL_ID)
    print("[INFO] HF_HOME:", hf_home.resolve())
    print("[INFO] Loading tokenizer only.")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        cache_dir=str(hf_home),
        use_fast=True,
        trust_remote_code=False,
    )

    size_after = directory_size(hf_home)

    report: dict[str, Any] = {
        "model_id": MODEL_ID,
        "tokenizer_class": (
            tokenizer.__class__.__name__
        ),
        "is_fast": bool(tokenizer.is_fast),
        "model_max_length": (
            tokenizer.model_max_length
        ),
        "vocabulary_size": (
            tokenizer.vocab_size
        ),
        "special_tokens_map": (
            tokenizer.special_tokens_map
        ),
        "special_token_ids": {
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "cls_token_id": tokenizer.cls_token_id,
            "sep_token_id": tokenizer.sep_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "unk_token_id": tokenizer.unk_token_id,
            "mask_token_id": tokenizer.mask_token_id,
        },
        "hf_home": str(hf_home.resolve()),
        "cache_size_before_bytes": size_before,
        "cache_size_after_bytes": size_after,
        "downloaded_bytes": max(
            0,
            size_after - size_before,
        ),
    }

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "[OK] Tokenizer class:",
        report["tokenizer_class"],
    )
    print("[OK] Fast tokenizer:", report["is_fast"])
    print(
        "[OK] Model max length:",
        report["model_max_length"],
    )
    print(
        "[OK] Vocabulary size:",
        report["vocabulary_size"],
    )
    print(
        "[OK] Cache growth:",
        report["downloaded_bytes"],
        "bytes",
    )
    print("[OK] Written:", REPORT_PATH.resolve())


if __name__ == "__main__":
    main()
