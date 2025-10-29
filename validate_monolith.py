#!/usr/bin/env python3
"""
Validate questionnaire_monolith.json
=====================================

Validates the monolithic questionnaire structure against all invariants.
"""

import json
import hashlib
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any


class MonolithValidator:
    """Validator for questionnaire_monolith.json"""
    
    # Expected canonical constants
    EXPECTED_POLICY_AREAS = 10
    EXPECTED_DIMENSIONS = 6
    EXPECTED_CLUSTERS = 4
    EXPECTED_MICRO_QUESTIONS = 300
    EXPECTED_MESO_QUESTIONS = 4
    EXPECTED_MACRO_QUESTIONS = 1
    EXPECTED_TOTAL_QUESTIONS = 305
    EXPECTED_BASE_SLOTS = 30
    QUESTIONS_PER_BASE_SLOT = 10
    
    CANONICAL_CLUSTERS = {
        'CLUSTER_1': ['P2', 'P3', 'P7'],
        'CLUSTER_2': ['P1', 'P5', 'P6'],
        'CLUSTER_3': ['P4', 'P8'],
        'CLUSTER_4': ['P9', 'P10']
    }
    
    CANONICAL_SCORING_MODALITIES = ['TYPE_A', 'TYPE_B', 'TYPE_C', 'TYPE_D', 'TYPE_E', 'TYPE_F']
    
    def __init__(self, monolith_path: str):
        self.monolith_path = Path(monolith_path)
        self.monolith = None
        self.errors = []
        self.warnings = []
        
    def error(self, message: str):
        """Record an error."""
        self.errors.append(message)
        print(f"❌ ERROR: {message}")
    
    def warning(self, message: str):
        """Record a warning."""
        self.warnings.append(message)
        print(f"⚠️  WARNING: {message}")
    
    def success(self, message: str):
        """Record a success."""
        print(f"✅ {message}")
    
    def validate(self) -> bool:
        """Run all validations. Returns True if all pass."""
        print("=" * 70)
        print(f"Validating: {self.monolith_path}")
        print("=" * 70)
        
        if not self.load_monolith():
            return False
        
        self.validate_structure()
        self.validate_question_counts()
        self.validate_base_slots()
        self.validate_clusters()
        self.validate_micro_questions()
        self.validate_meso_questions()
        self.validate_macro_question()
        self.validate_integrity_hash()
        
        print("\n" + "=" * 70)
        print("VALIDATION SUMMARY")
        print("=" * 70)
        
        if self.errors:
            print(f"\n❌ FAILED with {len(self.errors)} error(s):")
            for error in self.errors:
                print(f"  - {error}")
        
        if self.warnings:
            print(f"\n⚠️  {len(self.warnings)} warning(s):")
            for warning in self.warnings:
                print(f"  - {warning}")
        
        if not self.errors:
            print("\n✅ ALL VALIDATIONS PASSED")
            return True
        
        return False
    
    def load_monolith(self) -> bool:
        """Load and parse the monolith file."""
        if not self.monolith_path.exists():
            self.error(f"File not found: {self.monolith_path}")
            return False
        
        if self.monolith_path.stat().st_size == 0:
            self.error("File is empty")
            return False
        
        try:
            with open(self.monolith_path, 'r', encoding='utf-8') as f:
                self.monolith = json.load(f)
            self.success(f"Loaded monolith ({self.monolith_path.stat().st_size:,} bytes)")
            return True
        except json.JSONDecodeError as e:
            self.error(f"Invalid JSON: {e}")
            return False
    
    def validate_structure(self):
        """Validate top-level structure."""
        print("\n--- Structure Validation ---")
        
        required_keys = ['version', 'generated_at', 'integrity', 'blocks']
        for key in required_keys:
            if key not in self.monolith:
                self.error(f"Missing top-level key: {key}")
            else:
                self.success(f"Has top-level key: {key}")
        
        if 'blocks' in self.monolith:
            blocks = self.monolith['blocks']
            required_blocks = ['niveles_abstraccion', 'micro_questions', 'meso_questions', 'macro_question', 'scoring']
            for block in required_blocks:
                if block not in blocks:
                    self.error(f"Missing block: {block}")
                else:
                    self.success(f"Has block: {block}")
    
    def validate_question_counts(self):
        """Validate question counts."""
        print("\n--- Question Count Validation ---")
        
        blocks = self.monolith.get('blocks', {})
        
        micro_count = len(blocks.get('micro_questions', []))
        if micro_count == self.EXPECTED_MICRO_QUESTIONS:
            self.success(f"Micro questions: {micro_count}")
        else:
            self.error(f"Expected {self.EXPECTED_MICRO_QUESTIONS} micro questions, got {micro_count}")
        
        meso_count = len(blocks.get('meso_questions', []))
        if meso_count == self.EXPECTED_MESO_QUESTIONS:
            self.success(f"Meso questions: {meso_count}")
        else:
            self.error(f"Expected {self.EXPECTED_MESO_QUESTIONS} meso questions, got {meso_count}")
        
        macro = blocks.get('macro_question')
        if macro and isinstance(macro, dict):
            self.success("Macro question: 1")
        else:
            self.error("Missing or invalid macro question")
        
        total = micro_count + meso_count + (1 if macro else 0)
        if total == self.EXPECTED_TOTAL_QUESTIONS:
            self.success(f"Total questions: {total}")
        else:
            self.error(f"Expected {self.EXPECTED_TOTAL_QUESTIONS} total questions, got {total}")
    
    def validate_base_slots(self):
        """Validate base_slot distribution."""
        print("\n--- Base Slot Validation ---")
        
        micro_questions = self.monolith.get('blocks', {}).get('micro_questions', [])
        base_slot_counts = defaultdict(int)
        
        for q in micro_questions:
            base_slot = q.get('base_slot')
            if base_slot:
                base_slot_counts[base_slot] += 1
        
        if len(base_slot_counts) == self.EXPECTED_BASE_SLOTS:
            self.success(f"Base slots: {len(base_slot_counts)}")
        else:
            self.error(f"Expected {self.EXPECTED_BASE_SLOTS} base_slots, got {len(base_slot_counts)}")
        
        # Check each base_slot has exactly 10 questions
        mismatches = []
        for slot in sorted(base_slot_counts.keys()):
            count = base_slot_counts[slot]
            if count != self.QUESTIONS_PER_BASE_SLOT:
                mismatches.append(f"{slot}={count}")
        
        if mismatches:
            self.error(f"Base slots with incorrect count: {', '.join(mismatches)}")
        else:
            self.success(f"All base_slots have exactly {self.QUESTIONS_PER_BASE_SLOT} questions")
    
    def validate_clusters(self):
        """Validate cluster hermeticity."""
        print("\n--- Cluster Hermeticity Validation ---")
        
        niveles = self.monolith.get('blocks', {}).get('niveles_abstraccion', {})
        clusters = niveles.get('clusters', [])
        
        if len(clusters) == self.EXPECTED_CLUSTERS:
            self.success(f"Clusters: {len(clusters)}")
        else:
            self.error(f"Expected {self.EXPECTED_CLUSTERS} clusters, got {len(clusters)}")
        
        for i, cluster_def in enumerate(clusters, 1):
            cluster_id = f'CLUSTER_{i}'
            legacy_areas = cluster_def.get('legacy_policy_area_ids', [])
            expected = self.CANONICAL_CLUSTERS.get(cluster_id, [])
            
            if set(legacy_areas) == set(expected):
                self.success(f"{cluster_id}: {legacy_areas} ✓")
            else:
                self.error(f"{cluster_id}: expected {expected}, got {legacy_areas}")
    
    def validate_micro_questions(self):
        """Validate micro question structure."""
        print("\n--- Micro Question Structure Validation ---")
        
        micro_questions = self.monolith.get('blocks', {}).get('micro_questions', [])
        
        required_fields = [
            'question_global', 'question_id', 'base_slot', 'text',
            'scoring_modality', 'expected_elements', 'pattern_refs'
        ]
        
        missing_fields = defaultdict(list)
        empty_texts = []
        invalid_modalities = []
        
        for q in micro_questions:
            q_id = q.get('question_id', 'UNKNOWN')
            
            # Check required fields
            for field in required_fields:
                if field not in q:
                    missing_fields[field].append(q_id)
            
            # Check text not empty
            if not q.get('text', '').strip():
                empty_texts.append(q_id)
            
            # Check scoring_modality valid
            modality = q.get('scoring_modality')
            if modality and modality not in self.CANONICAL_SCORING_MODALITIES:
                invalid_modalities.append(f"{q_id}:{modality}")
        
        if missing_fields:
            for field, q_ids in missing_fields.items():
                self.error(f"Missing field '{field}' in {len(q_ids)} questions: {q_ids[:5]}")
        else:
            self.success("All micro questions have required fields")
        
        if empty_texts:
            self.error(f"Empty text in {len(empty_texts)} questions: {empty_texts[:5]}")
        else:
            self.success("All micro questions have non-empty text")
        
        if invalid_modalities:
            self.error(f"Invalid modalities: {invalid_modalities[:5]}")
        else:
            self.success("All micro questions have valid scoring_modality")
    
    def validate_meso_questions(self):
        """Validate meso question structure."""
        print("\n--- Meso Question Structure Validation ---")
        
        meso_questions = self.monolith.get('blocks', {}).get('meso_questions', [])
        
        for q in meso_questions:
            q_id = q.get('question_id', 'UNKNOWN')
            
            if q.get('type') != 'MESO':
                self.error(f"{q_id}: type should be 'MESO', got {q.get('type')}")
            
            if 'cluster_id' not in q:
                self.error(f"{q_id}: missing cluster_id")
            
            if not q.get('text'):
                self.error(f"{q_id}: missing text")
        
        if len(meso_questions) == self.EXPECTED_MESO_QUESTIONS:
            self.success("All meso questions valid")
    
    def validate_macro_question(self):
        """Validate macro question structure."""
        print("\n--- Macro Question Structure Validation ---")
        
        macro = self.monolith.get('blocks', {}).get('macro_question')
        
        if not macro:
            self.error("Missing macro question")
            return
        
        if macro.get('type') != 'MACRO':
            self.error(f"Macro type should be 'MACRO', got {macro.get('type')}")
        
        if macro.get('question_global') != 305:
            self.error(f"Macro question_global should be 305, got {macro.get('question_global')}")
        
        if 'fallback' not in macro:
            self.error("Macro question missing fallback")
        
        if not macro.get('text'):
            self.error("Macro question missing text")
        
        if all([
            macro.get('type') == 'MACRO',
            macro.get('question_global') == 305,
            'fallback' in macro,
            macro.get('text')
        ]):
            self.success("Macro question structure valid")
    
    def validate_integrity_hash(self):
        """Validate integrity hash."""
        print("\n--- Integrity Hash Validation ---")
        
        integrity = self.monolith.get('integrity', {})
        stored_hash = integrity.get('monolith_hash')
        
        if not stored_hash:
            self.error("Missing monolith_hash in integrity block")
            return
        
        self.success(f"Hash present: {stored_hash[:16]}...")
        
        # Validate question counts in integrity block
        counts = integrity.get('question_count', {})
        if counts.get('micro') == 300:
            self.success("Integrity: micro count = 300")
        else:
            self.error(f"Integrity: micro count should be 300, got {counts.get('micro')}")
        
        if counts.get('meso') == 4:
            self.success("Integrity: meso count = 4")
        else:
            self.error(f"Integrity: meso count should be 4, got {counts.get('meso')}")
        
        if counts.get('macro') == 1:
            self.success("Integrity: macro count = 1")
        else:
            self.error(f"Integrity: macro count should be 1, got {counts.get('macro')}")
        
        if counts.get('total') == 305:
            self.success("Integrity: total count = 305")
        else:
            self.error(f"Integrity: total count should be 305, got {counts.get('total')}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate questionnaire_monolith.json')
    parser.add_argument('monolith_file', help='Path to questionnaire_monolith.json')
    
    args = parser.parse_args()
    
    validator = MonolithValidator(args.monolith_file)
    success = validator.validate()
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
