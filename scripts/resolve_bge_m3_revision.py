"""Resolve the exact Hugging Face revision of BGE-M3."""

from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

MODEL_ID = "BAAI/bge-m3"

OUTPUT_PATH = Path(
    "outputs/bge_m3_model_revision.json"
)


def main() -> None:
    """Resolve and persist the exact model repository revision."""

    load_dotenv(
        dotenv_path=Path(".env"),
        override=False,
    )

    hf_home = os.getenv("HF_HOME")

    if not hf_home:
        raise RuntimeError(
            "HF_HOME is not configured in .env."
        )

    model_info = HfApi().model_info(
        repo_id=MODEL_ID,
        revision="main",
        files_metadata=True,
    )

    revision = model_info.sha

    if not revision:
        raise RuntimeError(
            "The Hugging Face Hub returned no model revision."
        )

    siblings = model_info.siblings or []

    files = [
        {
            "name": sibling.rfilename,
            "size": sibling.size,
        }
        for sibling in siblings
    ]

    report = {
        "model_id": MODEL_ID,
        "resolved_revision": revision,
        "hf_home": str(Path(hf_home).resolve()),
        "file_count": len(files),
        "files": files,
    }

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

    print("[OK] Model ID:", MODEL_ID)
    print("[OK] Revision:", revision)
    print("[OK] Repository files:", len(files))
    print("[OK] HF_HOME:", Path(hf_home).resolve())
    print("[OK] Written:", OUTPUT_PATH.resolve())


if __name__ == "__main__":
    main()
