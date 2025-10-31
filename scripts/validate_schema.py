#!/usr/bin/env python3
"""
Validate Schema Script

Validates the questionnaire monolith schema at initialization.
This script is designed to run in CI/CD pipelines to ensure schema
integrity before deployment.

Usage:
    python scripts/validate_schema.py [monolith_file] [--strict] [--report OUTPUT]

Exit codes:
    0 - Schema validation passed
    1 - Schema validation failed
"""

import sys
import json
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from validation.schema_validator import (
    MonolithSchemaValidator,
    SchemaInitializationError,
    validate_monolith_schema
)

# Try to import orchestrator, but make it optional
try:
    from orchestrator import get_questionnaire_provider
    HAS_ORCHESTRATOR = True
except ImportError:
    HAS_ORCHESTRATOR = False


def load_monolith(monolith_path: str = None):
    """
    Load monolith from file or orchestrator.
    
    Args:
        monolith_path: Optional path to monolith file
        
    Returns:
        dict: Monolith configuration
    """
    if monolith_path:
        with open(monolith_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        # Use orchestrator provider if available
        if not HAS_ORCHESTRATOR:
            raise ImportError(
                "Orchestrator module not available. Please provide a monolith file path."
            )
        provider = get_questionnaire_provider()
        return provider.load()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Validate questionnaire monolith schema'
    )
    parser.add_argument(
        'monolith_file',
        nargs='?',
        default=None,
        help='Path to monolith file (uses orchestrator default if not provided)'
    )
    parser.add_argument(
        '--strict',
        action='store_true',
        help='Enable strict mode (raise exception on failure)'
    )
    parser.add_argument(
        '--report',
        type=str,
        help='Path to save validation report JSON'
    )
    parser.add_argument(
        '--schema',
        type=str,
        help='Path to JSON schema file for validation'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("MONOLITH SCHEMA VALIDATION")
    print("=" * 70)
    print()
    
    try:
        # Load monolith
        print(f"Loading monolith...")
        if args.monolith_file:
            print(f"  Source: {args.monolith_file}")
        else:
            print(f"  Source: orchestrator default")
        
        monolith = load_monolith(args.monolith_file)
        print(f"  ✅ Loaded successfully")
        print()
        
        # Validate schema
        print(f"Validating schema...")
        validator = MonolithSchemaValidator(schema_path=args.schema)
        report = validator.validate_monolith(monolith, strict=args.strict)
        
        print()
        print("=" * 70)
        print("VALIDATION RESULTS")
        print("=" * 70)
        print()
        
        print(f"Schema version: {report.schema_version}")
        print(f"Timestamp: {report.timestamp}")
        print(f"Schema hash: {report.schema_hash[:16]}...")
        print()
        
        print(f"Question counts:")
        for level, count in report.question_counts.items():
            print(f"  {level}: {count}")
        print()
        
        print(f"Referential integrity:")
        for check, passed in report.referential_integrity.items():
            status = "✅" if passed else "❌"
            print(f"  {status} {check}: {'PASS' if passed else 'FAIL'}")
        print()
        
        if report.warnings:
            print(f"⚠️  Warnings ({len(report.warnings)}):")
            for warning in report.warnings:
                print(f"  - {warning}")
            print()
        
        if report.errors:
            print(f"❌ Errors ({len(report.errors)}):")
            for error in report.errors:
                print(f"  - {error}")
            print()
        
        # Save report if requested
        if args.report:
            report_path = Path(args.report)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report.model_dump(), f, indent=2, ensure_ascii=False)
            
            print(f"📄 Report saved to: {args.report}")
            print()
        
        print("=" * 70)
        
        if report.validation_passed:
            print("✅ SCHEMA VALIDATION PASSED")
            print("=" * 70)
            return 0
        else:
            print("❌ SCHEMA VALIDATION FAILED")
            print("=" * 70)
            return 1
    
    except SchemaInitializationError as e:
        print()
        print("=" * 70)
        print("❌ SCHEMA INITIALIZATION ERROR")
        print("=" * 70)
        print()
        print(str(e))
        print()
        return 1
    
    except Exception as e:
        print()
        print("=" * 70)
        print("❌ UNEXPECTED ERROR")
        print("=" * 70)
        print()
        print(f"Error: {e}")
        print()
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
