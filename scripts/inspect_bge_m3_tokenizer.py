"""Inspect the real BGE-M3 tokenizer and offset mappings."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from transformers import AutoTokenizer

from sah_rag.chunking import (
    HuggingFaceOffsetTokenizer,
    RegexOffsetTokenizer,
)

MODEL_ID = "BAAI/bge-m3"
OUTPUT_PATH = Path(
    "outputs/bge_m3_tokenizer_inspection.json"
)


SAMPLES = {
    "english": (
        "Article 5 establishes rights and obligations."
    ),
    "greek": (
        "Άρθρο 5: Δικαιώματα και υποχρεώσεις."
    ),
    "legal_citation": (
        "See 11 U.S.C. § 541(b)(1)."
    ),
    "mixed": (
        "Dense + sparse retrieval για το άρθρο 5."
    ),
    "unicode": (
        "Παράρτημα Α – εφαρμογή του Κανονισμού."
    ),
}


def main() -> None:
    """Inspect real BGE-M3 tokenization."""

    load_dotenv(
        dotenv_path=Path(".env"),
        override=False,
    )

    hf_home = os.getenv("HF_HOME")

    if not hf_home:
        raise RuntimeError(
            "HF_HOME is not configured."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        cache_dir=hf_home,
        use_fast=True,
        trust_remote_code=False,
        local_files_only=True,
    )

    adapter = HuggingFaceOffsetTokenizer(
        tokenizer,
        model_id=MODEL_ID,
    )

    regex_tokenizer = RegexOffsetTokenizer()

    report: dict[str, Any] = {
        "model_id": MODEL_ID,
        "tokenizer_id": adapter.tokenizer_id,
        "tokenizer_class": (
            tokenizer.__class__.__name__
        ),
        "is_fast": tokenizer.is_fast,
        "model_max_length": (
            adapter.model_max_length
        ),
        "special_tokens_count": (
            adapter.special_tokens_count
        ),
        "samples": {},
    }

    for sample_name, text in SAMPLES.items():
        bge_tokens = adapter.tokenize_with_offsets(text)
        regex_tokens = (
            regex_tokenizer.tokenize_with_offsets(text)
        )

        encoded_with_specials = tokenizer(
            text,
            add_special_tokens=True,
            truncation=False,
        )

        report["samples"][sample_name] = {
            "text": text,
            "bge_content_token_count": len(bge_tokens),
            "bge_model_input_token_count": len(
                encoded_with_specials["input_ids"]
            ),
            "regex_token_count": len(regex_tokens),
            "bge_tokens": [
                {
                    "index": token.token_index,
                    "text": token.text,
                    "start": token.start,
                    "end": token.end,
                }
                for token in bge_tokens
            ],
        }

        print("=" * 72)
        print("SAMPLE:", sample_name)
        print("TEXT:", text)
        print(
            "BGE content tokens:",
            len(bge_tokens),
        )
        print(
            "BGE model-input tokens:",
            len(
                encoded_with_specials["input_ids"]
            ),
        )
        print(
            "Regex tokens:",
            len(regex_tokens),
        )
        print(
            "BGE slices:",
            [
                token.text
                for token in bge_tokens
            ],
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("[OK] Tokenizer ID:", adapter.tokenizer_id)
    print(
        "[OK] Special tokens:",
        adapter.special_tokens_count,
    )
    print(
        "[OK] Model max length:",
        adapter.model_max_length,
    )
    print("[OK] Written:", OUTPUT_PATH.resolve())


if __name__ == "__main__":
    main()
