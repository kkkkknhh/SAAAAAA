#!/usr/bin/env python3
"""
Example: Validating questionnaire_monolith.json against its JSON Schema

This example demonstrates how to:
1. Load the questionnaire monolith and its schema
2. Validate the monolith structure
3. Extract specific information from validated data
4. Handle validation errors gracefully
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

try:
    import jsonschema
    from jsonschema import Draft7Validator, ValidationError
except ImportError:
    print("Error: jsonschema not installed. Install with: pip install jsonschema")
    sys.exit(1)


def load_monolith_and_schema() -> tuple[Dict, Dict]:
    """Load the questionnaire monolith and its schema."""
    repo_root = Path(__file__).parent.parent
    
    monolith_path = repo_root / "data" / "questionnaire_monolith.json"
    schema_path = repo_root / "config" / "schemas" / "questionnaire_monolith.schema.json"
    
    with open(monolith_path) as f:
        monolith = json.load(f)
    
    with open(schema_path) as f:
        schema = json.load(f)
    
    return monolith, schema


def validate_monolith(monolith: Dict, schema: Dict) -> tuple[bool, List[str]]:
    """
    Validate monolith against schema.
    
    Returns:
        Tuple of (is_valid, list_of_error_messages)
    """
    validator = Draft7Validator(schema)
    errors = []
    
    for error in validator.iter_errors(monolith):
        path = ".".join(str(p) for p in error.path) if error.path else "root"
        errors.append(f"[{path}] {error.message}")
    
    return len(errors) == 0, errors


def get_question_by_id(monolith: Dict, question_id: str) -> Optional[Dict]:
    """
    Get a specific question by its ID from the monolith.
    
    Args:
        monolith: The validated monolith data
        question_id: Question ID (e.g., 'Q001', 'MESO_1', 'MACRO_1')
    
    Returns:
        Question dict or None if not found
    """
    # Check micro questions
    for q in monolith['blocks']['micro_questions']:
        if q['question_id'] == question_id:
            return q
    
    # Check meso questions
    for q in monolith['blocks']['meso_questions']:
        if q['question_id'] == question_id:
            return q
    
    # Check macro question
    macro = monolith['blocks']['macro_question']
    if macro['question_id'] == question_id:
        return macro
    
    return None


def get_questions_by_dimension(monolith: Dict, dimension_id: str) -> List[Dict]:
    """
    Get all questions for a specific dimension.
    
    Args:
        monolith: The validated monolith data
        dimension_id: Dimension ID (e.g., 'DIM01', 'DIM02')
    
    Returns:
        List of question dicts
    """
    questions = []
    for q in monolith['blocks']['micro_questions']:
        if q['dimension_id'] == dimension_id:
            questions.append(q)
    return questions


def get_questions_by_policy_area(monolith: Dict, policy_area_id: str) -> List[Dict]:
    """
    Get all questions for a specific policy area.
    
    Args:
        monolith: The validated monolith data
        policy_area_id: Policy area ID (e.g., 'PA01', 'PA02')
    
    Returns:
        List of question dicts
    """
    questions = []
    for q in monolith['blocks']['micro_questions']:
        if q['policy_area_id'] == policy_area_id:
            questions.append(q)
    return questions


def get_questions_by_cluster(monolith: Dict, cluster_id: str) -> List[Dict]:
    """
    Get all questions for a specific cluster.
    
    Args:
        monolith: The validated monolith data
        cluster_id: Cluster ID (e.g., 'CL01', 'CL02')
    
    Returns:
        List of question dicts
    """
    questions = []
    for q in monolith['blocks']['micro_questions']:
        if q['cluster_id'] == cluster_id:
            questions.append(q)
    return questions


def get_scoring_modality(monolith: Dict, modality_type: str) -> Optional[Dict]:
    """
    Get scoring modality definition.
    
    Args:
        monolith: The validated monolith data
        modality_type: Type of scoring modality (e.g., 'TYPE_A', 'TYPE_B')
    
    Returns:
        Modality definition dict or None if not found
    """
    return monolith['blocks']['scoring']['modality_definitions'].get(modality_type)


def print_monolith_stats(monolith: Dict):
    """Print statistics about the monolith."""
    print("\n" + "=" * 80)
    print("Questionnaire Monolith Statistics")
    print("=" * 80)
    
    # Version info
    print(f"\nVersion: {monolith['version']}")
    print(f"Schema Version: {monolith['schema_version']}")
    print(f"Generated: {monolith['generated_at']}")
    
    # Question counts
    counts = monolith['integrity']['question_count']
    print(f"\nQuestion Counts:")
    print(f"  Micro:  {counts['micro']}")
    print(f"  Meso:   {counts['meso']}")
    print(f"  Macro:  {counts['macro']}")
    print(f"  Total:  {counts['total']}")
    
    # Dimensions
    dimensions = monolith['blocks']['niveles_abstraccion']['dimensions']
    print(f"\nDimensions: {len(dimensions)}")
    for dim in dimensions:
        label = dim['i18n']['keys']['label_es']
        print(f"  {dim['dimension_id']}: {label}")
    
    # Clusters
    clusters = monolith['blocks']['niveles_abstraccion']['clusters']
    print(f"\nClusters: {len(clusters)}")
    for cluster in clusters:
        label = cluster['i18n']['keys']['label_es']
        pas = ", ".join(cluster['policy_area_ids'])
        print(f"  {cluster['cluster_id']}: {label}")
        print(f"    Policy Areas: {pas}")
    
    # Scoring modalities
    modalities = monolith['blocks']['scoring']['modalities']
    print(f"\nScoring Modalities: {len(modalities)}")
    for mod_type, mod_def in modalities.items():
        print(f"  {mod_type}: {mod_def['description']}")
    
    print("\n" + "=" * 80)


def main():
    """Main example function."""
    print("Loading questionnaire monolith and schema...")
    monolith, schema = load_monolith_and_schema()
    
    print("Validating monolith structure...")
    is_valid, errors = validate_monolith(monolith, schema)
    
    if not is_valid:
        print(f"\n✗ Validation failed with {len(errors)} errors:")
        for error in errors[:10]:  # Show first 10 errors
            print(f"  {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")
        return 1
    
    print("✓ Validation successful!\n")
    
    # Print statistics
    print_monolith_stats(monolith)
    
    # Example: Get a specific question
    print("\nExample 1: Get question by ID")
    print("-" * 80)
    question = get_question_by_id(monolith, 'Q001')
    if question:
        print(f"Question ID: {question['question_id']}")
        print(f"Global Number: {question['question_global']}")
        print(f"Text: {question['text'][:80]}...")
        print(f"Scoring Modality: {question['scoring_modality']}")
        print(f"Number of Patterns: {len(question['patterns'])}")
    
    # Example: Get questions by dimension
    print("\nExample 2: Get questions by dimension")
    print("-" * 80)
    dim_questions = get_questions_by_dimension(monolith, 'DIM01')
    print(f"Questions in DIM01 (Insumos): {len(dim_questions)}")
    print(f"Question IDs: {', '.join(q['question_id'] for q in dim_questions[:5])}...")
    
    # Example: Get questions by policy area
    print("\nExample 3: Get questions by policy area")
    print("-" * 80)
    pa_questions = get_questions_by_policy_area(monolith, 'PA01')
    print(f"Questions in PA01 (Derechos de las mujeres): {len(pa_questions)}")
    
    # Example: Get questions by cluster
    print("\nExample 4: Get questions by cluster")
    print("-" * 80)
    cluster_questions = get_questions_by_cluster(monolith, 'CL02')
    print(f"Questions in CL02 (Grupos Poblacionales): {len(cluster_questions)}")
    
    # Example: Get scoring modality
    print("\nExample 5: Get scoring modality definition")
    print("-" * 80)
    modality = get_scoring_modality(monolith, 'TYPE_A')
    if modality:
        print(f"TYPE_A Definition:")
        print(f"  Description: {modality['description']}")
        print(f"  Aggregation: {modality['aggregation']}")
        print(f"  Failure Code: {modality['failure_code']}")
    
    print("\n" + "=" * 80)
    print("✓ All examples completed successfully!")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
