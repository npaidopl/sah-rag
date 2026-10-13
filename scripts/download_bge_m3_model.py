"""Download the exact BGE-M3 model snapshot."""

from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import snapshot_download

MODEL_ID = "BAAI/bge-m3"

REVISION_REPORT = Path(
    "outputs/bge_m3_model_revision.json"
)

OUTPUT_REPORT = Path(
    "outputs/bge_m3_model_download.json"
)


def directory_size(path: Path) -> int:
    """Return recursive file size."""

    if not path.exists():
        return 0

    return sum(
        item.stat().st_size
        for item in path.rglob("*")
        if item.is_file()
    )


def main() -> None:
    """Download the pinned model snapshot."""

    load_dotenv(
        dotenv_path=Path(".env"),
        override=False,
    )

    hf_home_raw = os.getenv("HF_HOME")

    if not hf_home_raw:
        raise RuntimeError(
            "HF_HOME is not configured in .env."
        )

    revision_data = json.loads(
        REVISION_REPORT.read_text(
            encoding="utf-8"
        )
    )

    revision = revision_data["resolved_revision"]
    hf_home = Path(hf_home_raw)

    size_before = directory_size(hf_home)

    print("[INFO] Model ID:", MODEL_ID)
    print("[INFO] Revision:", revision)
    print("[INFO] HF_HOME:", hf_home.resolve())
    print("[INFO] Downloading pinned model snapshot.")

    snapshot_path = snapshot_download(
        repo_id=MODEL_ID,
        revision=revision,
        cache_dir=str(hf_home),
    )

    size_after = directory_size(hf_home)

    report = {
        "model_id": MODEL_ID,
        "revision": revision,
        "snapshot_path": str(
            Path(snapshot_path).resolve()
        ),
        "cache_size_before_bytes": size_before,
        "cache_size_after_bytes": size_after,
        "downloaded_bytes": max(
            0,
            size_after - size_before,
        ),
    }

    OUTPUT_REPORT.write_text(
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
        "[OK] Snapshot:",
        Path(snapshot_path).resolve(),
    )
    print(
        "[OK] Downloaded:",
        report["downloaded_bytes"],
        "bytes",
    )
    print("[OK] Written:", OUTPUT_REPORT.resolve())


if __name__ == "__main__":
    main()
