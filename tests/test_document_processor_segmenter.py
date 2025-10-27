import hashlib
import json
from pathlib import Path

import pytest

from Analyzer_one import DocumentProcessor, CanonicalQuestionSegmenter

try:
    import jsonschema
except ImportError:  # pragma: no cover - jsonschema is an optional dependency
    jsonschema = None


def _serializable_payload(segmentation: dict) -> dict:
    """Convert segmentation payload into JSON-serializable structure."""

    question_segments = {
        "|".join(key_tuple): payload
        for key_tuple, payload in segmentation["question_segments"].items()
    }

    return {
        "metadata": segmentation["metadata"],
        "question_segments": question_segments,
        "question_segment_index": segmentation["question_segment_index"],
    }


def test_canonical_contract_loading_ensures_schema_alignment():
    contracts, questionnaire_meta, rubric_meta, contracts_hash = (
        DocumentProcessor.load_canonical_question_contracts()
    )

    assert len(contracts) == 300
    assert questionnaire_meta.get("version") is not None
    assert rubric_meta.get("version") is not None
    assert isinstance(contracts_hash, str) and len(contracts_hash) == 64

    canonical_keys = {
        (contract.canonical_question_id, contract.policy_area_id, contract.dimension_id)
        for contract in contracts
    }
    assert len(canonical_keys) == len(contracts) == 300

    first = contracts[0]
    assert first.canonical_question_id == "Q001"
    assert first.legacy_question_id.count("-") == 2
    assert first.policy_area_id.startswith("PA")
    assert first.dimension_id.startswith("DIM")
    assert len(first.contract_hash) == 64
    assert isinstance(first.expected_elements, list)
    assert isinstance(first.search_patterns, dict)
    assert isinstance(first.verification_patterns, list)


@pytest.fixture(scope="module")
def canonical_segmenter():
    return CanonicalQuestionSegmenter()


def test_segmenter_deterministic_and_attested(canonical_segmenter):
    plan_text = """
    PLAN DE DESARROLLO MUNICIPAL 2024-2027
    El diagnóstico establece una línea base cuantitativa del 2023 con datos del DANE.
    Se reporta un porcentaje del 42.5% y la fuente es la Encuesta Nacional.
    """

    first_run = canonical_segmenter.segment_plan(plan_text)
    second_run = canonical_segmenter.segment_plan(plan_text)

    assert first_run["metadata"] == second_run["metadata"]
    assert first_run["question_segments"] == second_run["question_segments"]
    assert first_run["question_segment_index"] == second_run["question_segment_index"]

    metadata = first_run["metadata"]
    expected_hash = hashlib.sha256(plan_text.encode("utf-8")).hexdigest()
    assert metadata["input_sha256"] == expected_hash
    assert metadata["contracts_sha256"] == canonical_segmenter.contracts_hash

    assert 0 < metadata["coverage_ratio"] <= 1

    expected_order = [
        (
            contract.canonical_question_id,
            contract.policy_area_id,
            contract.dimension_id,
        )
        for contract in canonical_segmenter.contracts
    ]
    assert list(first_run["question_segments"].keys()) == expected_order

    matched_questions = [
        segment for segment in first_run["question_segments"].values()
        if segment["evidence_manifest"]["matched"]
    ]
    assert matched_questions, "Expected at least one matched question for sample plan"

    for matched in matched_questions:
        manifest = matched["evidence_manifest"]
        assert manifest["matched_segment_count"] > 0
        assert isinstance(manifest["expected_elements"], list)
        assert isinstance(manifest["matched_segments"], list)
        assert manifest["attestation"]["contract_sha256"] == matched["contract_hash"]
        for segment in manifest["matched_segments"]:
            assert len(segment["segment_hash"]) == 64
            assert segment["matched_patterns"], "Matched segment should list patterns"

    for index_entry in first_run["question_segment_index"]:
        key_tuple = tuple(index_entry["key_tuple"])
        assert key_tuple in first_run["question_segments"]
        payload = first_run["question_segments"][key_tuple]
        assert payload["contract_hash"] == index_entry["contract_hash"]
        assert payload["evidence_manifest"] == index_entry["evidence_manifest"]

    if jsonschema is not None:
        schema_path = Path("schemas/question_segmentation.schema.json")
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        jsonschema.validate(_serializable_payload(first_run), schema)


def test_negative_control_zero_coverage(canonical_segmenter):
    noise_plan = "Solo texto genérico sin patrones relevantes ni datos verificables."
    segmentation = canonical_segmenter.segment_plan(noise_plan)

    metadata = segmentation["metadata"]
    assert metadata["coverage_ratio"] < 0.05
    assert metadata["covered_contracts"] <= 15

    sample_keys = list(segmentation["question_segments"].keys())[:5]
    for key in sample_keys:
        manifest = segmentation["question_segments"][key]["evidence_manifest"]
        assert manifest["matched"] is False
        assert manifest["matched_segment_count"] == 0
        assert manifest["pattern_hits"] == {}

    matched_entries = sum(
        1 for entry in segmentation["question_segment_index"] if entry["evidence_manifest"]["matched"]
    )
    assert matched_entries == metadata["covered_contracts"]
    if metadata["total_contracts"]:
        assert pytest.approx(metadata["coverage_ratio"], rel=1e-6) == (
            matched_entries / metadata["total_contracts"]
        )
