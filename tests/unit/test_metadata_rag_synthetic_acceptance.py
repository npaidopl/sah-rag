"""Tests for the Metadata-RAG synthetic acceptance run."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = (
    REPO_ROOT
    / "processed"
    / "canonical"
    / "infrastructure-pilot-synthetic-v1"
)
DATASET_ID = "infrastructure-pilot-synthetic-v1"

SCRIPT_PATH = (
    REPO_ROOT / "scripts" / "run_metadata_rag_synthetic_acceptance.py"
)

# Import the module dynamically so the test can call run_acceptance
# without adding scripts/ to sys.path globally.
sys.path.insert(0, str(SCRIPT_PATH.parent))
import run_metadata_rag_synthetic_acceptance as acceptance  # type: ignore[import-not-found]  # noqa: E402

REQUIRED_GATES: tuple[str, ...] = (
    "canonical_input_accepted",
    "preprocessing_completed",
    "query_count_matches",
    "ranked_results_nonempty",
    "canonical_ids_recovered",
    "duplicate_ids_preserved",
    "every_query_top2_matches_ground_truth",
    "fallback_not_triggered_on_fixture",
    "evaluation_case_created",
    "generation_disabled",
    "dataset_id_consistent",
    "state_reopened",
    "repeat_run_deterministic_or_classified",
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- Full run --------------------------------------------------------------


def test_acceptance_run_succeeds(tmp_path: Path) -> None:
    manifest = acceptance.run_acceptance(
        fixture_root=FIXTURE_ROOT,
        output_dir=tmp_path / "out",
        dataset_id=DATASET_ID,
    )
    assert manifest["all_gates_passed"] is True
    assert manifest["encoder_kind"] == (
        acceptance.ENCODER_KIND_DETERMINISTIC_CONTRACT
    )
    assert manifest["document_count"] == 12
    assert manifest["query_count"] == 6
    assert manifest["index_size"] == 12
    for gate in REQUIRED_GATES:
        assert manifest["gates"][gate] is True, gate


def test_all_expected_artifacts_written(tmp_path: Path) -> None:
    out = tmp_path / "out"
    acceptance.run_acceptance(
        fixture_root=FIXTURE_ROOT,
        output_dir=out,
        dataset_id=DATASET_ID,
    )
    for rel in (
        "manifest.json",
        "readiness.json",
        "checksums.json",
        "document_embeddings.jsonl",
        "query_embeddings.jsonl",
        "retrieval_results.jsonl",
        "expanded_documents.jsonl",
        "evaluation_cases.jsonl",
        "state-v1/documents.jsonl",
        "state-v1/manifest.json",
        "state-v1/checksums.json",
    ):
        assert (out / rel).is_file(), rel


def test_checksums_match_recorded_hashes(tmp_path: Path) -> None:
    out = tmp_path / "out"
    acceptance.run_acceptance(
        fixture_root=FIXTURE_ROOT,
        output_dir=out,
        dataset_id=DATASET_ID,
    )
    checksums = json.loads(
        (out / "checksums.json").read_text(encoding="utf-8")
    )
    for rel, expected in checksums.items():
        assert _sha256_file(out / rel) == expected, rel


# --- Artifact shapes -------------------------------------------------------


def test_document_embeddings_shape(tmp_path: Path) -> None:
    out = tmp_path / "out"
    acceptance.run_acceptance(
        fixture_root=FIXTURE_ROOT,
        output_dir=out,
        dataset_id=DATASET_ID,
    )
    lines = (out / "document_embeddings.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(lines) == 12
    for line in lines:
        obj = json.loads(line)
        assert obj["record_kind"] == "document_embedding"
        assert obj["dense_dimension"] == 1024
        assert obj["dense_normalized"] is True
        assert obj["model_id"] == "BAAI/bge-m3"


def test_query_embeddings_do_not_store_text(tmp_path: Path) -> None:
    out = tmp_path / "out"
    acceptance.run_acceptance(
        fixture_root=FIXTURE_ROOT,
        output_dir=out,
        dataset_id=DATASET_ID,
    )
    text = (out / "query_embeddings.jsonl").read_text(encoding="utf-8")
    for line in text.splitlines():
        obj = json.loads(line)
        assert "query_text" not in obj
        assert "query_text_sha256" in obj


def test_retrieval_results_shape(tmp_path: Path) -> None:
    out = tmp_path / "out"
    acceptance.run_acceptance(
        fixture_root=FIXTURE_ROOT,
        output_dir=out,
        dataset_id=DATASET_ID,
    )
    lines = (out / "retrieval_results.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(lines) == 6
    for line in lines:
        obj = json.loads(line)
        assert obj["ranked_items"], obj
        for item in obj["ranked_items"]:
            assert item["score_semantics"] == "HIGHER_IS_BETTER"
            assert item["canonical_doc_id"].startswith("pilot-doc-")
            assert item["metadata"]["fallback_flags"][
                "metadata_match_none"
            ] is False


def test_top2_matches_ground_truth_per_query(tmp_path: Path) -> None:
    out = tmp_path / "out"
    acceptance.run_acceptance(
        fixture_root=FIXTURE_ROOT,
        output_dir=out,
        dataset_id=DATASET_ID,
    )
    results = {}
    for line in (out / "retrieval_results.jsonl").read_text(
        encoding="utf-8"
    ).splitlines():
        obj = json.loads(line)
        results[obj["query_id"]] = [
            it["canonical_doc_id"] for it in obj["ranked_items"]
        ]

    expected = {}
    for line in (FIXTURE_ROOT / "examples.jsonl").read_text(
        encoding="utf-8"
    ).splitlines():
        obj = json.loads(line)
        expected[obj["query"]["query_id"]] = obj["ground_truth"][
            "relevant_ids"
        ]

    for query_id, gt in expected.items():
        top2 = results[query_id][:2]
        assert top2 == gt, f"{query_id}: got {top2}, expected {gt}"


def test_evaluation_cases_shape(tmp_path: Path) -> None:
    out = tmp_path / "out"
    acceptance.run_acceptance(
        fixture_root=FIXTURE_ROOT,
        output_dir=out,
        dataset_id=DATASET_ID,
    )
    lines = (out / "evaluation_cases.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(lines) == 6
    for line in lines:
        obj = json.loads(line)
        assert obj["query_id"].startswith("pilot-query-")
        assert obj["ranked_doc_ids"]
        assert obj["relevant_doc_ids"]


def test_readiness_json_shape(tmp_path: Path) -> None:
    out = tmp_path / "out"
    acceptance.run_acceptance(
        fixture_root=FIXTURE_ROOT,
        output_dir=out,
        dataset_id=DATASET_ID,
    )
    readiness = json.loads(
        (out / "readiness.json").read_text(encoding="utf-8")
    )
    assert readiness["synthetic_acceptance_passed"] is True
    for gate in REQUIRED_GATES:
        assert readiness["gates"][gate] is True, gate


# --- Determinism and safety ------------------------------------------------


def test_repeat_run_produces_identical_retrieval_bytes(
    tmp_path: Path,
) -> None:
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    acceptance.run_acceptance(
        fixture_root=FIXTURE_ROOT,
        output_dir=out_a,
        dataset_id=DATASET_ID,
    )
    acceptance.run_acceptance(
        fixture_root=FIXTURE_ROOT,
        output_dir=out_b,
        dataset_id=DATASET_ID,
    )
    for rel in (
        "retrieval_results.jsonl",
        "expanded_documents.jsonl",
        "evaluation_cases.jsonl",
        "document_embeddings.jsonl",
        "query_embeddings.jsonl",
    ):
        assert (out_a / rel).read_bytes() == (out_b / rel).read_bytes(), rel


def test_refuses_non_empty_output_dir(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / "existing.txt").write_text("x", encoding="utf-8")
    with pytest.raises(
        acceptance.MetadataRAGSyntheticAcceptanceError,
        match="already contains",
    ):
        acceptance.run_acceptance(
            fixture_root=FIXTURE_ROOT,
            output_dir=out,
            dataset_id=DATASET_ID,
        )


def test_force_flag_recreates_output_dir(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / "existing.txt").write_text("x", encoding="utf-8")
    manifest = acceptance.run_acceptance(
        fixture_root=FIXTURE_ROOT,
        output_dir=out,
        dataset_id=DATASET_ID,
        force=True,
    )
    assert manifest["all_gates_passed"] is True
    assert not (out / "existing.txt").exists()


# --- CLI --------------------------------------------------------------------


def test_cli_runs_to_completion(tmp_path: Path) -> None:
    out = tmp_path / "cli-out"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--fixture-root",
            str(FIXTURE_ROOT),
            "--output-dir",
            str(out),
            "--dataset-id",
            DATASET_ID,
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "[OK] Metadata-RAG synthetic acceptance" in result.stdout
    assert "[ PASS ] every_query_top2_matches_ground_truth" in (
        result.stdout
    )


# --- Reference to the frozen spec -----------------------------------------


def test_encoder_kind_is_not_a_language_model(tmp_path: Path) -> None:
    """The acceptance run must label the encoder as a contract encoder."""
    out = tmp_path / "out"
    manifest = acceptance.run_acceptance(
        fixture_root=FIXTURE_ROOT,
        output_dir=out,
        dataset_id=DATASET_ID,
    )
    assert manifest["encoder_kind"] == (
        "DETERMINISTIC_CONTRACT_ENCODER"
    )
    # Sanity: the frozen spec still demands BGE-M3 for real runs.
    spec = yaml.safe_load(
        (
            REPO_ROOT
            / "baseline_audit"
            / "metadata_rag_clean_room_spec.yaml"
        ).read_text(encoding="utf-8")
    )
    assert (
        spec["semantic_retrieval"]["embedder"]["model_id"]
        == "BAAI/bge-m3"
    )
