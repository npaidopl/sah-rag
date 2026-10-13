"""Tests for rag_system.baselines.metadata_rag_metadata."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest
import yaml

from rag_system.baselines.metadata_rag_metadata import (
    CAPITALIZED_PHRASE_PATTERN,
    EXTRACTOR_VERSION,
    MEDIUM_MAX_WORDS,
    NUMERIC_TOKEN_PATTERN,
    SCHEMA_VERSION,
    SHORT_MAX_WORDS,
    YEAR_MENTION_PATTERN,
    MetadataRecord,
    build_document_metadata,
    build_query_metadata,
    compute_length_bucket,
    compute_word_count,
    derive_query_constraints,
    extract_capitalized_phrases,
    extract_numeric_tokens,
    extract_year_mentions,
)
from rag_system.canonical.loaders import iter_validated_models
from rag_system.canonical.models import (
    CanonicalDocument,
    CanonicalExample,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = (
    REPO_ROOT / "baseline_audit" / "metadata_rag_clean_room_spec.yaml"
)
AMENDMENT_PATH = (
    REPO_ROOT
    / "baseline_audit"
    / "metadata_rag_clean_room_spec_amendment_01.yaml"
)
FIXTURE_ROOT = (
    REPO_ROOT
    / "processed"
    / "canonical"
    / "infrastructure-pilot-synthetic-v1"
)


def _load_fixture_documents() -> tuple[CanonicalDocument, ...]:
    """Load repo-local synthetic fixture documents.

    Deliberately bypasses get_canonical_root(): the test-local fixture
    lives in the repository under processed/canonical/, not under the
    external canonical suite root.
    """
    return tuple(
        iter_validated_models(
            FIXTURE_ROOT / "documents.jsonl", CanonicalDocument
        )
    )


def _load_fixture_examples() -> tuple[CanonicalExample, ...]:
    return tuple(
        iter_validated_models(
            FIXTURE_ROOT / "examples.jsonl", CanonicalExample
        )
    )


# --- Frozen-spec alignment -------------------------------------------------


def test_hardcoded_patterns_match_amendment() -> None:
    amendment = yaml.safe_load(AMENDMENT_PATH.read_text(encoding="utf-8"))
    patterns: dict[str, str] = amendment["amendment"]["patch"][
        "metadata_schema"
    ]["derived_tier"]["frozen_patterns"]
    assert patterns["capitalized_phrase"] == CAPITALIZED_PHRASE_PATTERN
    assert patterns["year_mention"] == YEAR_MENTION_PATTERN
    assert patterns["numeric_token"] == NUMERIC_TOKEN_PATTERN


def test_amendment_binds_to_frozen_spec_sha256() -> None:
    amendment: dict[str, Any] = yaml.safe_load(
        AMENDMENT_PATH.read_text(encoding="utf-8")
    )
    spec_sha = hashlib.sha256(SPEC_PATH.read_bytes()).hexdigest().upper()
    assert amendment["amendment"]["applies_to"]["sha256"] == spec_sha


# --- Extraction primitives -------------------------------------------------


def test_capitalized_phrases_exclude_sentence_initial_article() -> None:
    assert extract_capitalized_phrases(
        "The Amber Observatory records data."
    ) == ("amber observatory",)


def test_capitalized_phrases_exclude_single_token_capitals() -> None:
    assert extract_capitalized_phrases("Northbridge is quiet today.") == ()


def test_capitalized_phrases_are_lowercased_sorted_deduplicated() -> None:
    text = (
        "Visit the Silver Workshop and the Winter Navigation School, "
        "then the Silver Workshop again."
    )
    assert extract_capitalized_phrases(text) == (
        "silver workshop",
        "winter navigation school",
    )


def test_capitalized_phrases_symmetric_document_and_query() -> None:
    doc = "The Amber Observatory records atmospheric patterns."
    query = "Which instruments record patterns at the Amber Observatory?"
    doc_phrases = extract_capitalized_phrases(doc)
    query_phrases = extract_capitalized_phrases(query)
    assert doc_phrases == query_phrases == ("amber observatory",)


def test_year_mentions_sorted_and_deduplicated() -> None:
    text = "In 2019, 2020, and 2019. Also 1999 and 2000."
    assert extract_year_mentions(text) == (1999, 2000, 2019, 2020)


def test_year_mentions_ignore_non_year_numerics() -> None:
    assert extract_year_mentions("There were 42 items in 1899.") == ()


def test_numeric_tokens_preserve_string_form_and_dedupe() -> None:
    text = "Values 1.0, 2, 3.14, 1.0, and 42."
    assert extract_numeric_tokens(text) == ("1.0", "2", "3.14", "42")


def test_word_count_matches_whitespace_split() -> None:
    assert compute_word_count("a b c") == 3
    assert compute_word_count("  a\t b \n c  ") == 3
    assert compute_word_count("") == 0


def test_length_bucket_thresholds() -> None:
    assert compute_length_bucket(0) == "short"
    assert compute_length_bucket(SHORT_MAX_WORDS) == "short"
    assert compute_length_bucket(SHORT_MAX_WORDS + 1) == "medium"
    assert compute_length_bucket(MEDIUM_MAX_WORDS) == "medium"
    assert compute_length_bucket(MEDIUM_MAX_WORDS + 1) == "long"


# --- Model construction ----------------------------------------------------


def test_build_document_metadata_full_shape() -> None:
    rec = build_document_metadata(
        dataset="infrastructure-pilot-synthetic-v1",
        doc_id="pilot-doc-01",
        source_dataset_id="infrastructure-pilot-synthetic-v1",
        text_normalized="The Amber Observatory records atmospheric patterns.",
        source_content_group_id="pilot-content-group-01",
        title="Amber  Observatory  ",
    )
    assert rec.source_kind == "document"
    assert rec.doc_id == "pilot-doc-01"
    assert rec.title == "amber  observatory"
    assert rec.capitalized_phrases == ("amber observatory",)
    assert rec.schema_version == SCHEMA_VERSION
    assert rec.extractor_version == EXTRACTOR_VERSION


def test_build_query_metadata_has_no_provenance_tier() -> None:
    rec = build_query_metadata(
        dataset="infrastructure-pilot-synthetic-v1",
        query_id="pilot-query-01",
        query_text=(
            "Which instruments record patterns at the Amber Observatory?"
        ),
    )
    assert rec.source_kind == "query"
    assert rec.doc_id is None
    assert rec.source_dataset_id is None
    assert rec.source_content_group_id is None
    assert rec.title is None
    assert rec.capitalized_phrases == ("amber observatory",)


def test_document_metadata_requires_provenance() -> None:
    with pytest.raises(ValueError):
        MetadataRecord(
            source_kind="document",
            dataset="x",
            capitalized_phrases=(),
            year_mentions=(),
            numeric_tokens=(),
            char_length=0,
            word_count=0,
            length_bucket="short",
        )


def test_builders_reject_empty_inputs() -> None:
    with pytest.raises(ValueError):
        build_document_metadata(
            dataset="",
            doc_id="x",
            source_dataset_id="x",
            text_normalized="text",
        )
    with pytest.raises(ValueError):
        build_query_metadata(dataset="x", query_id="y", query_text="")


# --- Query constraint derivation -------------------------------------------


def test_derive_query_constraints_from_query_record() -> None:
    rec = build_query_metadata(
        dataset="x",
        query_id="q-1",
        query_text="In 2019, the Amber Observatory recorded value 42.",
    )
    constraints = derive_query_constraints(rec)
    ids = {c.constraint_id for c in constraints}
    assert "cap:amber observatory" in ids
    assert "year:2019" in ids
    assert "num:42" in ids
    for c in constraints:
        assert c.provenance == "QUERY_TEXT"


def test_derive_query_constraints_rejects_document_record() -> None:
    rec = build_document_metadata(
        dataset="x",
        doc_id="d-1",
        source_dataset_id="x",
        text_normalized="Some text.",
    )
    with pytest.raises(ValueError):
        derive_query_constraints(rec)


# --- Fixture-level acceptance ----------------------------------------------


def test_fixture_files_are_present() -> None:
    assert (FIXTURE_ROOT / "documents.jsonl").is_file()
    assert (FIXTURE_ROOT / "examples.jsonl").is_file()


def test_fixture_document_and_example_counts() -> None:
    docs = _load_fixture_documents()
    examples = _load_fixture_examples()
    assert len(docs) == 12
    assert len(examples) == 6


def test_fixture_every_document_has_a_capitalized_phrase() -> None:
    docs = _load_fixture_documents()
    for doc in docs:
        phrases = extract_capitalized_phrases(doc.text_normalized)
        assert phrases, f"{doc.doc_id} has no capitalized phrases"


def test_fixture_query_filter_selects_exactly_ground_truth() -> None:
    docs = _load_fixture_documents()
    examples = _load_fixture_examples()

    doc_metadata: dict[str, Any] = {}
    for d in docs:
        source_dataset_id = str(
            d.metadata.get("source_dataset_id", d.dataset)
        )
        source_content_group_id_raw = str(
            d.metadata.get("source_content_group_id", "")
        )
        doc_metadata[d.doc_id] = build_document_metadata(
            dataset=d.dataset,
            doc_id=d.doc_id,
            source_dataset_id=source_dataset_id,
            text_normalized=d.text_normalized,
            source_content_group_id=source_content_group_id_raw or None,
            title=d.title,
        )

    for example in examples:
        qm = build_query_metadata(
            dataset=example.query.dataset,
            query_id=example.query.query_id,
            query_text=example.query.question,
        )
        query_phrases = set(qm.capitalized_phrases)
        assert query_phrases, example.query.query_id

        matching = {
            doc_id
            for doc_id, rec in doc_metadata.items()
            if query_phrases & set(rec.capitalized_phrases)
        }
        expected = set(example.ground_truth.relevant_ids)
        assert matching == expected, (
            f"{example.query.query_id}: matched {sorted(matching)} "
            f"but expected {sorted(expected)}"
        )
