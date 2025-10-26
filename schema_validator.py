# -*- coding: utf-8 -*-
"""Data contract validator for questionnaire and rubric configurations."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

try:
    import jsonschema
    from jsonschema import ValidationError  # noqa: F401
except ImportError:  # pragma: no cover - jsonschema shipped with repo tooling
    jsonschema = None


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT

QUESTIONNAIRE_PATH = REPO_ROOT / "questionnaire.json"
RUBRIC_PATH = REPO_ROOT / "rubric_scoring.json"
CHECKSUM_PATH = REPO_ROOT / "config" / "metadata_checksums.json"
QUESTIONNAIRE_SCHEMA = REPO_ROOT / "schemas" / "questionnaire.schema.json"
RUBRIC_SCHEMA = REPO_ROOT / "schemas" / "rubric_scoring.schema.json"

FLOAT_TOLERANCE = 1e-6


@dataclass
class SchemaValidationReport:
    """Structured validation report."""

    is_valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def add_error(self, message: str) -> None:
        self.errors.append(message)
        self.is_valid = False

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)


class SchemaValidator:
    """Validate questionnaire and rubric data contracts."""

    def __init__(self) -> None:
        self.questionnaire_schema = self._load_schema(QUESTIONNAIRE_SCHEMA)
        self.rubric_schema = self._load_schema(RUBRIC_SCHEMA)

    @staticmethod
    def _load_schema(path: Path) -> Optional[Dict[str, Any]]:
        try:
            with path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except FileNotFoundError:
            return None

    @staticmethod
    def _canonical_hash(payload: Mapping[str, Any]) -> str:
        clone = json.loads(json.dumps(payload, ensure_ascii=False))
        clone = dict(clone)
        clone.pop("content_hash", None)
        canonical = json.dumps(clone, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _validate_with_schema(
        self,
        payload: Mapping[str, Any],
        schema: Optional[Mapping[str, Any]],
        report: SchemaValidationReport,
        name: str,
    ) -> None:
        if not jsonschema or not schema:
            report.add_warning(f"jsonschema missing – skipped schema validation for {name}")
            return
        try:
            jsonschema.validate(instance=payload, schema=schema)
        except Exception as exc:  # pragma: no cover - jsonschema raises specialised errors
            report.add_error(f"{name} schema violation: {exc}")

    def _load_payload(self, path: Path, report: SchemaValidationReport) -> Optional[Dict[str, Any]]:
        try:
            with path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except Exception as exc:
            report.add_error(f"Unable to read {path.name}: {exc}")
            return None

    def validate_questionnaire(self, path: Path = QUESTIONNAIRE_PATH) -> Tuple[SchemaValidationReport, Optional[Dict[str, Any]]]:
        report = SchemaValidationReport()
        payload = self._load_payload(path, report)
        if not payload:
            return report, None

        self._validate_with_schema(payload, self.questionnaire_schema, report, "questionnaire.json")

        expected_hash = payload.get("content_hash")
        actual_hash = self._canonical_hash(payload)
        if expected_hash != actual_hash:
            report.add_error(
                f"questionnaire.json content_hash mismatch (expected {expected_hash}, computed {actual_hash})"
            )

        metadata = payload.get("metadata", {})
        sources = {item["key"] for item in metadata.get("sources_of_verification", [])}
        if len(sources) != len(metadata.get("sources_of_verification", [])):
            report.add_error("Duplicate keys in sources_of_verification")

        policy_areas = metadata.get("policy_areas", [])
        if len(policy_areas) != 10:
            report.add_error(f"Expected 10 policy areas, found {len(policy_areas)}")
        policy_area_ids = {pa["policy_area_id"] for pa in policy_areas if "policy_area_id" in pa}
        if len(policy_area_ids) != len(policy_areas):
            report.add_error("Duplicated policy_area_id detected")

        dimensions = metadata.get("dimensions", [])
        if len(dimensions) != 6:
            report.add_error(f"Expected 6 dimensions, found {len(dimensions)}")
        dimension_ids = {dim["dimension_id"] for dim in dimensions if "dimension_id" in dim}

        cluster_map: Dict[str, str] = {}
        for cluster in metadata.get("clusters", []):
            for pa_id in cluster.get("policy_area_ids", []):
                cluster_map[pa_id] = cluster["cluster_id"]

        questions = payload.get("questions", [])
        if len(questions) != 300:
            report.add_error(f"Expected 300 questions, found {len(questions)}")

        question_ids = set()
        for question in questions:
            qid = question.get("question_id")
            if qid in question_ids:
                report.add_error(f"Duplicated question_id {qid}")
            question_ids.add(qid)

            pa_id = question.get("policy_area_id")
            if pa_id not in policy_area_ids:
                report.add_error(f"Question {qid} references unknown policy_area_id {pa_id}")

            dim_id = question.get("dimension_id")
            if dim_id not in dimension_ids:
                report.add_error(f"Question {qid} references unknown dimension_id {dim_id}")

            expected_cluster = cluster_map.get(pa_id)
            if expected_cluster and question.get("cluster_id") != expected_cluster:
                report.add_error(
                    f"Question {qid} cluster mismatch: expected {expected_cluster} from policy area {pa_id}"
                )

            missing_keys = set(question.get("required_evidence_keys", [])) - sources
            if missing_keys:
                report.add_error(f"Question {qid} references unknown evidence keys {sorted(missing_keys)}")

        return report, payload

    def validate_rubric(
        self,
        questionnaire_payload: Optional[Dict[str, Any]],
        path: Path = RUBRIC_PATH,
    ) -> SchemaValidationReport:
        report = SchemaValidationReport()
        payload = self._load_payload(path, report)
        if not payload:
            return report

        self._validate_with_schema(payload, self.rubric_schema, report, "rubric_scoring.json")

        expected_hash = payload.get("content_hash")
        actual_hash = self._canonical_hash(payload)
        if expected_hash != actual_hash:
            report.add_error(
                f"rubric_scoring.json content_hash mismatch (expected {expected_hash}, computed {actual_hash})"
            )

        if questionnaire_payload:
            q_version = questionnaire_payload.get("version")
            r_requires = payload.get("requires_questionnaire_version")
            if q_version != r_requires:
                report.add_error(
                    f"Version compatibility mismatch: questionnaire {q_version} vs rubric requires {r_requires}"
                )

        # Weight validations
        aggregation = payload.get("aggregation", {})
        scoring_modalities = payload.get("scoring_modalities", {})
        if "TYPE_F" in scoring_modalities and "determinism" not in scoring_modalities["TYPE_F"]:
            report.add_error("TYPE_F modality must declare determinism metadata")
        self._validate_weight_matrix(aggregation.get("dimension_question_weights", {}), "dimension", report)
        self._validate_weight_matrix(aggregation.get("policy_area_dimension_weights", {}), "policy_area", report)
        self._validate_weight_matrix(aggregation.get("cluster_policy_area_weights", {}), "cluster", report)
        macro_weights = aggregation.get("macro_cluster_weights", {})
        if not self._weights_sum_to_one(macro_weights):
            report.add_error("macro_cluster_weights must sum to 1.0")

        rubric_matrix = payload.get("rubric_matrix", {})
        required_keys = payload.get("required_evidence_keys", {})

        if questionnaire_payload:
            questions = questionnaire_payload.get("questions", [])
            for question in questions:
                pa_id = question.get("policy_area_id")
                dim_id = question.get("dimension_id")
                qid = question.get("question_id")
                modality = question.get("scoring_modality")
                matrix_entry = rubric_matrix.get(pa_id, {}).get(dim_id)
                if not matrix_entry:
                    report.add_error(f"Rubric matrix missing entry for {pa_id}/{dim_id}")
                    continue
                if modality not in matrix_entry.get("allowed_modalities", []):
                    report.add_error(
                        f"Question {qid} modality {modality} not allowed by rubric matrix for {pa_id}/{dim_id}"
                    )
                matrix_keys = set(matrix_entry.get("required_evidence_keys", []))
                question_keys = set(question.get("required_evidence_keys", []))
                if not question_keys.issubset(matrix_keys):
                    report.add_error(
                        f"Question {qid} requires evidence keys {sorted(question_keys)} not covered by matrix"
                    )
                pa_keys = set(required_keys.get(pa_id, []))
                if not question_keys.issubset(pa_keys):
                    report.add_error(
                        f"Question {qid} evidence keys not present in rubric required_evidence_keys for {pa_id}"
                    )

            # Ensure dimension_question_weights cover all questions
            dimension_weights = aggregation.get("dimension_question_weights", {})
            missing_weights = [
                q["question_id"]
                for q in questions
                if q.get("question_id") not in dimension_weights.get(q.get("dimension_id"), {})
            ]
            if missing_weights:
                report.add_error(f"Questions missing dimension weights: {missing_weights[:5]}...")

        # NA rules coverage
        modalities = payload.get("scoring_modalities", {})
        na_modalities = set(payload.get("na_rules", {}).get("modalities", {}).keys())
        missing_modalities = set(modalities.keys()) - na_modalities
        if missing_modalities:
            report.add_error(f"NA rules missing for modalities: {sorted(missing_modalities)}")

        required_pa_keys = set(required_keys.keys())
        if questionnaire_payload:
            pa_ids = {pa["policy_area_id"] for pa in questionnaire_payload.get("metadata", {}).get("policy_areas", [])}
            missing_pa = pa_ids - required_pa_keys
            if missing_pa:
                report.add_error(f"Rubric required_evidence_keys missing policy areas: {sorted(missing_pa)}")

        return report

    def _validate_weight_matrix(
        self,
        matrix: Mapping[str, Mapping[str, float]],
        scope: str,
        report: SchemaValidationReport,
    ) -> None:
        for parent, weights in matrix.items():
            if not self._weights_sum_to_one(weights):
                report.add_error(f"Weights for {scope} {parent} must sum to 1.0")

    @staticmethod
    def _weights_sum_to_one(weights: Mapping[str, float]) -> bool:
        total = sum(weights.values())
        return abs(total - 1.0) <= FLOAT_TOLERANCE

    def validate_metadata_checksums(self) -> SchemaValidationReport:
        report = SchemaValidationReport()
        if not CHECKSUM_PATH.exists():
            report.add_error("config/metadata_checksums.json missing")
            return report
        try:
            with CHECKSUM_PATH.open("r", encoding="utf-8") as handle:
                recorded = json.load(handle)
        except Exception as exc:
            report.add_error(f"Unable to read metadata checksums: {exc}")
            return report

        questionnaire_payload = self._load_payload(QUESTIONNAIRE_PATH, report)
        rubric_payload = self._load_payload(RUBRIC_PATH, report)

        if questionnaire_payload:
            expected = recorded.get("questionnaire.json")
            actual = self._canonical_hash(questionnaire_payload)
            if expected != actual:
                report.add_error(
                    f"Checksum mismatch for questionnaire.json (expected {expected}, computed {actual})"
                )
        if rubric_payload:
            expected = recorded.get("rubric_scoring.json")
            actual = self._canonical_hash(rubric_payload)
            if expected != actual:
                report.add_error(
                    f"Checksum mismatch for rubric_scoring.json (expected {expected}, computed {actual})"
                )
        execution_path = REPO_ROOT / "execution_mapping.yaml"
        if execution_path.exists():
            expected = recorded.get("execution_mapping.yaml")
            with execution_path.open("r", encoding="utf-8") as handle:
                text = "\n".join(line.rstrip() for line in handle.read().splitlines()).strip() + "\n"
            actual = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if expected != actual:
                report.add_error(
                    f"Checksum mismatch for execution_mapping.yaml (expected {expected}, computed {actual})"
                )
        else:
            report.add_warning("execution_mapping.yaml missing during checksum validation")

        return report

    def validate_all(self) -> Tuple[SchemaValidationReport, SchemaValidationReport, SchemaValidationReport]:
        questionnaire_report, questionnaire_payload = self.validate_questionnaire()
        rubric_report = self.validate_rubric(questionnaire_payload)
        checksum_report = self.validate_metadata_checksums()
        return questionnaire_report, rubric_report, checksum_report


def run() -> int:
    validator = SchemaValidator()
    q_report, r_report, c_report = validator.validate_all()

    failed = False
    for name, report in (
        ("Questionnaire", q_report),
        ("Rubric", r_report),
        ("Metadata Checksums", c_report),
    ):
        if report.errors:
            failed = True
            print(f"❌ {name} validation failed:")
            for error in report.errors:
                print(f"  - {error}")
        else:
            print(f"✅ {name} validation passed")
        for warning in report.warnings:
            print(f"⚠️  {name}: {warning}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(run())
