#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Schema Validator - JSON Schema Validation for Canonical Configuration
=====================================================================

Validates questionnaire.json and rubric_scoring.json against their schemas
at orchestrator startup and before each run.

Validation Rules:
- All PAxx referenced by CLxx must exist
- Every Qxxx maps to exactly one PAxx and one DIMxx
- 100% question coverage by PAxx
- No orphaned PAxx
- Version compatibility between files

Author: Integration Team
Version: 2.0.0
Python: 3.10+
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field

try:
    import jsonschema
    from jsonschema import validate, ValidationError
except ImportError:
    jsonschema = None
    ValidationError = Exception

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class SchemaValidationReport:
    """Report of schema validation with detailed errors."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_error(self, error: str):
        """Add validation error."""
        self.errors.append(error)
        self.is_valid = False
    
    def add_warning(self, warning: str):
        """Add validation warning."""
        self.warnings.append(warning)
    
    def summary(self) -> str:
        """Generate validation summary."""
        if self.is_valid:
            status = "✓ PASSED"
        else:
            status = "✗ FAILED"
        
        return (f"Schema Validation {status}: "
                f"{len(self.errors)} errors, {len(self.warnings)} warnings")


class SchemaValidator:
    """
    Validates questionnaire.json and rubric_scoring.json against schemas.
    
    Enforces:
    - JSON Schema compliance
    - Referential integrity (PAxx, DIMxx, Qxxx)
    - Coverage completeness
    - Version compatibility
    """
    
    def __init__(
        self,
        questionnaire_schema_path: str = "schemas/questionnaire.schema.json",
        rubric_schema_path: str = "schemas/rubric_scoring.schema.json"
    ):
        """
        Initialize schema validator.
        
        Args:
            questionnaire_schema_path: Path to questionnaire schema
            rubric_schema_path: Path to rubric scoring schema
        """
        self.questionnaire_schema = self._load_schema(questionnaire_schema_path)
        self.rubric_schema = self._load_schema(rubric_schema_path)
        
        logger.info("SchemaValidator initialized")
    
    def _load_schema(self, path: str) -> Optional[Dict[str, Any]]:
        """Load JSON schema from file."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                schema = json.load(f)
            logger.info(f"✓ Loaded schema: {path}")
            return schema
        except FileNotFoundError:
            logger.error(f"Schema file not found: {path}")
            return None
        except Exception as e:
            logger.error(f"Failed to load schema {path}: {e}")
            return None
    
    def validate_questionnaire(
        self,
        questionnaire_path: str
    ) -> SchemaValidationReport:
        """
        Validate questionnaire.json against schema and business rules.
        
        Args:
            questionnaire_path: Path to questionnaire.json
            
        Returns:
            SchemaValidationReport with validation results
        """
        report = SchemaValidationReport(is_valid=True)
        
        logger.info("=" * 80)
        logger.info("VALIDATING QUESTIONNAIRE.JSON")
        logger.info("=" * 80)
        
        # Load questionnaire
        try:
            with open(questionnaire_path, 'r', encoding='utf-8') as f:
                questionnaire = json.load(f)
        except Exception as e:
            report.add_error(f"Failed to load questionnaire: {e}")
            return report
        
        # 1. JSON Schema validation
        if jsonschema and self.questionnaire_schema:
            try:
                validate(instance=questionnaire, schema=self.questionnaire_schema)
                logger.info("✓ JSON Schema validation passed")
            except ValidationError as e:
                report.add_error(f"Schema validation failed: {e.message}")
                logger.error(f"✗ Schema validation failed: {e.message}")
        else:
            report.add_warning("jsonschema not available, skipping schema validation")
        
        # 2. Extract metadata
        metadata = questionnaire.get("metadata", {})
        clusters = metadata.get("clusters", [])
        policy_areas = metadata.get("policy_areas", [])
        dimensions = metadata.get("dimensions", [])
        questions = questionnaire.get("questions", [])
        
        report.metadata["version"] = metadata.get("version", "UNKNOWN")
        report.metadata["cluster_count"] = len(clusters)
        report.metadata["policy_area_count"] = len(policy_areas)
        report.metadata["dimension_count"] = len(dimensions)
        report.metadata["question_count"] = len(questions)
        
        # 3. Validate cluster count (must be exactly 4)
        if len(clusters) != 4:
            report.add_error(f"Must have exactly 4 clusters, found {len(clusters)}")
        else:
            logger.info(f"✓ Cluster count: {len(clusters)}")
        
        # 4. Validate policy area count (must be exactly 10)
        if len(policy_areas) != 10:
            report.add_error(f"Must have exactly 10 policy areas, found {len(policy_areas)}")
        else:
            logger.info(f"✓ Policy area count: {len(policy_areas)}")
        
        # 5. Validate dimension count (must be exactly 6)
        if len(dimensions) != 6:
            report.add_error(f"Must have exactly 6 dimensions, found {len(dimensions)}")
        else:
            logger.info(f"✓ Dimension count: {len(dimensions)}")
        
        # 6. Check all PAxx referenced by CLxx exist
        pa_ids = {pa["policy_area_id"] for pa in policy_areas}
        
        for cluster in clusters:
            cluster_id = cluster.get("cluster_id")
            referenced_pas = cluster.get("policy_area_ids", [])
            
            for pa_id in referenced_pas:
                if pa_id not in pa_ids:
                    report.add_error(
                        f"Cluster {cluster_id} references non-existent PA: {pa_id}"
                    )
        
        if not report.errors:
            logger.info(f"✓ All cluster references valid")
        
        # 7. Check every Qxxx maps to exactly one PAxx and one DIMxx
        dim_ids = {dim["dimension_id"] for dim in dimensions}
        
        for question in questions:
            q_id = question.get("question_id", "UNKNOWN")
            pa_id = question.get("policy_area_id")
            dim_id = question.get("dimension_id")
            
            if pa_id not in pa_ids:
                report.add_error(f"Question {q_id} references non-existent PA: {pa_id}")
            
            if dim_id not in dim_ids:
                report.add_error(f"Question {q_id} references non-existent DIM: {dim_id}")
        
        if not report.errors:
            logger.info(f"✓ All question references valid")
        
        # 8. Check 100% coverage: all PAxx have at least one question
        questions_by_pa = {pa_id: 0 for pa_id in pa_ids}
        
        for question in questions:
            pa_id = question.get("policy_area_id")
            if pa_id in questions_by_pa:
                questions_by_pa[pa_id] += 1
        
        orphaned_pas = [pa_id for pa_id, count in questions_by_pa.items() if count == 0]
        
        if orphaned_pas:
            report.add_error(f"Orphaned policy areas (no questions): {orphaned_pas}")
        else:
            logger.info(f"✓ 100% policy area coverage")
        
        # 9. Report statistics
        logger.info(f"  Questions per PA: {dict(questions_by_pa)}")
        
        logger.info("=" * 80)
        logger.info(report.summary())
        logger.info("=" * 80)
        
        return report
    
    def validate_rubric_scoring(
        self,
        rubric_path: str,
        questionnaire_data: Optional[Dict[str, Any]] = None
    ) -> SchemaValidationReport:
        """
        Validate rubric_scoring.json against schema.
        
        Args:
            rubric_path: Path to rubric_scoring.json
            questionnaire_data: Optional questionnaire data for cross-validation
            
        Returns:
            SchemaValidationReport with validation results
        """
        report = SchemaValidationReport(is_valid=True)
        
        logger.info("=" * 80)
        logger.info("VALIDATING RUBRIC_SCORING.JSON")
        logger.info("=" * 80)
        
        # Load rubric
        try:
            with open(rubric_path, 'r', encoding='utf-8') as f:
                rubric = json.load(f)
        except Exception as e:
            report.add_error(f"Failed to load rubric: {e}")
            return report
        
        # 1. JSON Schema validation
        if jsonschema and self.rubric_schema:
            try:
                validate(instance=rubric, schema=self.rubric_schema)
                logger.info("✓ JSON Schema validation passed")
            except ValidationError as e:
                report.add_error(f"Schema validation failed: {e.message}")
                logger.error(f"✗ Schema validation failed: {e.message}")
        else:
            report.add_warning("jsonschema not available, skipping schema validation")
        
        # 2. Check version compatibility
        rubric_version = rubric.get("metadata", {}).get("version", "UNKNOWN")
        compatible_version = rubric.get("metadata", {}).get("compatible_questionnaire_version", "UNKNOWN")
        
        report.metadata["version"] = rubric_version
        report.metadata["compatible_questionnaire_version"] = compatible_version
        
        logger.info(f"  Rubric version: {rubric_version}")
        logger.info(f"  Compatible with questionnaire: {compatible_version}")
        
        # 3. Cross-validate with questionnaire if provided
        if questionnaire_data:
            metadata = questionnaire_data.get("metadata", {})
            q_version = metadata.get("version", "UNKNOWN")
            
            if q_version != compatible_version:
                report.add_warning(
                    f"Version mismatch: questionnaire {q_version} vs "
                    f"rubric expects {compatible_version}"
                )
        
        logger.info("=" * 80)
        logger.info(report.summary())
        logger.info("=" * 80)
        
        return report
    
    def validate_all(
        self,
        questionnaire_path: str,
        rubric_path: str
    ) -> Tuple[SchemaValidationReport, SchemaValidationReport]:
        """
        Validate both questionnaire and rubric with cross-validation.
        
        Args:
            questionnaire_path: Path to questionnaire.json
            rubric_path: Path to rubric_scoring.json
            
        Returns:
            Tuple of (questionnaire_report, rubric_report)
        """
        logger.info("\n" + "=" * 80)
        logger.info("COMPREHENSIVE SCHEMA VALIDATION")
        logger.info("=" * 80)
        
        # Validate questionnaire
        q_report = self.validate_questionnaire(questionnaire_path)
        
        # Load questionnaire for cross-validation
        questionnaire_data = None
        if q_report.is_valid:
            try:
                with open(questionnaire_path, 'r', encoding='utf-8') as f:
                    questionnaire_data = json.load(f)
            except:
                pass
        
        # Validate rubric
        r_report = self.validate_rubric_scoring(rubric_path, questionnaire_data)
        
        # Overall summary
        logger.info("\n" + "=" * 80)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Questionnaire: {q_report.summary()}")
        logger.info(f"Rubric: {r_report.summary()}")
        
        overall_valid = q_report.is_valid and r_report.is_valid
        logger.info(f"\nOverall Status: {'✓ ALL VALID' if overall_valid else '✗ ERRORS FOUND'}")
        logger.info("=" * 80)
        
        return q_report, r_report


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def example_usage():
    """Example validation usage."""
    validator = SchemaValidator()
    
    # Validate both files
    q_report, r_report = validator.validate_all(
        questionnaire_path="questionnaire.json",
        rubric_path="rubric_scoring.json"
    )
    
    if not q_report.is_valid:
        print("\n❌ Questionnaire validation FAILED:")
        for error in q_report.errors:
            print(f"  - {error}")
    
    if not r_report.is_valid:
        print("\n❌ Rubric validation FAILED:")
        for error in r_report.errors:
            print(f"  - {error}")


if __name__ == "__main__":
    example_usage()
