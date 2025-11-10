#!/usr/bin/env python3
"""Detect Embedded Calibrations - Find Methods with Local Parametrization

This script implements the requirement to identify and track methods with
embedded/inline calibration (local parametrization) as transitional anomalies.

Per the directive:
- Methods with local/in-script parametrization must be explicitly listed
- Each must be linked to file, location, and parametrization pattern
- This appendix is the authoritative backlog for migration
- No embedded calibrations can remain invisible

This tool detects:
1. Hard-coded thresholds, weights, and parameters
2. Magic numbers used for scoring/evaluation
3. Local configuration dictionaries
4. Inline parameter definitions

Output:
- Embedded calibration registry (JSON)
- Migration appendix with file locations
- Pattern analysis for each case
"""

import ast
import json
import re
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional


@dataclass
class EmbeddedCalibration:
    """Metadata for a method with embedded calibration."""
    
    method_id: str  # From canonical catalog
    canonical_name: str
    file_path: str
    line_number: int
    class_name: Optional[str]
    method_name: str
    
    # Embedded calibration details
    pattern_type: str  # "threshold", "weight", "config_dict", "magic_number"
    parameters_found: List[Dict[str, Any]]  # [{name, value, line}]
    parametrization_snippet: str  # Code snippet showing the parametrization
    
    # Migration tracking
    migration_priority: str  # "critical", "high", "medium", "low"
    migration_complexity: str  # "simple", "moderate", "complex"
    notes: str


class EmbeddedCalibrationDetector:
    """Detector for methods with embedded calibration."""
    
    # Patterns that indicate embedded calibration
    CALIBRATION_VARIABLE_PATTERNS = [
        r'\b(threshold|min|max|weight|alpha|beta|gamma|penalty|tolerance|sensitivity)\s*=\s*[\d\.]',
        r'\bscore_\w+\s*=\s*[\d\.]',
        r'\b(min|max)_\w+\s*=\s*[\d\.]',
        r'\b\w+_(weight|threshold|min|max|penalty)\s*=\s*[\d\.]',
    ]
    
    MAGIC_NUMBER_THRESHOLD = 2  # If method has more than 2 magic numbers, investigate
    
    def __init__(self, catalog_path: Path):
        """Initialize with canonical catalog."""
        with open(catalog_path) as f:
            self.catalog = json.load(f)
        
        self.embedded_calibrations: List[EmbeddedCalibration] = []
        self.repo_root = Path(catalog_path).parent.parent
    
    def detect_all(self) -> List[EmbeddedCalibration]:
        """Detect all embedded calibrations in methods that require calibration."""
        print("Detecting embedded calibrations...")
        
        # Focus on methods that require calibration but aren't centralized
        candidates = self.catalog['calibration_tracking']['unknown']
        
        print(f"  Analyzing {len(candidates)} methods marked as 'unknown' calibration status")
        
        for method in candidates:
            embedded = self._detect_in_method(method)
            if embedded:
                self.embedded_calibrations.append(embedded)
        
        print(f"  Found {len(self.embedded_calibrations)} methods with embedded calibrations")
        return self.embedded_calibrations
    
    def _detect_in_method(self, method: Dict[str, Any]) -> Optional[EmbeddedCalibration]:
        """Detect if a method has embedded calibration."""
        file_path = self.repo_root / method['file_path']
        
        if not file_path.exists():
            return None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()
            
            tree = ast.parse(source)
            
            # Find the method node
            method_node = self._find_method_node(
                tree, method['method_name'], method['class_name']
            )
            
            if not method_node:
                return None
            
            # Extract method source
            try:
                method_source = ast.unparse(method_node)
            except Exception:
                # Fallback to line-based extraction
                lines = source.split('\n')
                start = method['line_number'] - 1
                # Try to get ~20 lines
                end = min(start + 20, len(lines))
                method_source = '\n'.join(lines[start:end])
            
            # Detect calibration patterns
            patterns_found = self._detect_patterns(method_source, method_node)
            
            if not patterns_found['parameters']:
                return None
            
            # Determine migration priority
            priority = self._determine_priority(method, patterns_found)
            complexity = self._determine_complexity(patterns_found)
            
            # Extract snippet
            snippet = self._extract_snippet(method_source, patterns_found)
            
            return EmbeddedCalibration(
                method_id=method['unique_id'],
                canonical_name=method['canonical_name'],
                file_path=method['file_path'],
                line_number=method['line_number'],
                class_name=method['class_name'],
                method_name=method['method_name'],
                pattern_type=patterns_found['type'],
                parameters_found=patterns_found['parameters'],
                parametrization_snippet=snippet,
                migration_priority=priority,
                migration_complexity=complexity,
                notes=patterns_found['notes']
            )
        
        except Exception as e:
            print(f"    Warning: Could not analyze {method['canonical_name']}: {e}")
            return None
    
    def _find_method_node(self, tree: ast.Module, method_name: str,
                         class_name: Optional[str]) -> Optional[ast.FunctionDef]:
        """Find AST node for the method."""
        if class_name:
            # Find class first
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == class_name:
                    # Find method in class
                    for item in node.body:
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            if item.name == method_name:
                                return item
        else:
            # Module-level function
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name == method_name:
                        return node
        
        return None
    
    def _detect_patterns(self, source: str, node: ast.FunctionDef) -> Dict[str, Any]:
        """Detect calibration patterns in method source."""
        parameters = []
        pattern_type = "unknown"
        notes = []
        
        # 1. Check for explicit calibration variable patterns
        for pattern in self.CALIBRATION_VARIABLE_PATTERNS:
            matches = re.finditer(pattern, source, re.IGNORECASE)
            for match in matches:
                param_line = match.group(0)
                # Extract variable name and value
                parts = param_line.split('=')
                if len(parts) == 2:
                    var_name = parts[0].strip()
                    value = parts[1].strip()
                    parameters.append({
                        'name': var_name,
                        'value': value,
                        'line': param_line,
                        'type': 'explicit_variable'
                    })
        
        # 2. Check for config/param dictionaries
        dict_pattern = r'\b(config|params?|calibration|settings)\s*=\s*\{'
        if re.search(dict_pattern, source, re.IGNORECASE):
            pattern_type = "config_dict"
            notes.append("Contains configuration dictionary")
        
        # 3. Check for magic numbers in comparisons/operations
        # Look for numeric literals in the AST
        magic_numbers = []
        for subnode in ast.walk(node):
            if isinstance(subnode, ast.Num):
                # Ignore common non-magic numbers
                if subnode.n not in [0, 1, 2, -1, 0.0, 1.0]:
                    magic_numbers.append(subnode.n)
            elif isinstance(subnode, ast.Constant):
                if isinstance(subnode.value, (int, float)):
                    if subnode.value not in [0, 1, 2, -1, 0.0, 1.0]:
                        magic_numbers.append(subnode.value)
        
        if len(magic_numbers) >= self.MAGIC_NUMBER_THRESHOLD:
            for num in magic_numbers[:10]:  # Limit to first 10
                parameters.append({
                    'name': 'magic_number',
                    'value': num,
                    'line': f'numeric literal: {num}',
                    'type': 'magic_number'
                })
            if pattern_type == "unknown":
                pattern_type = "magic_numbers"
            notes.append(f"Contains {len(magic_numbers)} magic numbers")
        
        # 4. Determine primary pattern type
        if parameters:
            explicit_vars = [p for p in parameters if p['type'] == 'explicit_variable']
            if explicit_vars:
                pattern_type = "explicit_parameters"
            elif pattern_type == "unknown":
                pattern_type = "magic_numbers"
        
        return {
            'type': pattern_type,
            'parameters': parameters,
            'notes': '; '.join(notes) if notes else 'Embedded parameters detected'
        }
    
    def _determine_priority(self, method: Dict[str, Any], patterns: Dict[str, Any]) -> str:
        """Determine migration priority."""
        # Critical: Executors and scoring methods
        if 'executor' in method['layer'].lower() or 'scoring' in method['canonical_name'].lower():
            return "critical"
        
        # High: Analyzers with many parameters
        if method['layer'] == 'analyzer' and len(patterns['parameters']) > 5:
            return "high"
        
        # Medium: Other analyzers and processors
        if method['layer'] in ['analyzer', 'processor']:
            return "medium"
        
        # Low: Utilities and others
        return "low"
    
    def _determine_complexity(self, patterns: Dict[str, Any]) -> str:
        """Determine migration complexity."""
        param_count = len(patterns['parameters'])
        
        if param_count > 10:
            return "complex"
        elif param_count > 5:
            return "moderate"
        else:
            return "simple"
    
    def _extract_snippet(self, source: str, patterns: Dict[str, Any]) -> str:
        """Extract relevant code snippet."""
        # Get first 500 chars or up to first parameter
        snippet_lines = []
        for line in source.split('\n')[:20]:  # First 20 lines
            snippet_lines.append(line)
            # Stop if we've captured some parameters
            if len(snippet_lines) > 5 and any(p['name'] in line for p in patterns['parameters'][:3]):
                break
        
        snippet = '\n'.join(snippet_lines)
        if len(snippet) > 500:
            snippet = snippet[:500] + "..."
        
        return snippet


def generate_migration_appendix(embedded_calibrations: List[EmbeddedCalibration],
                               output_path: Path):
    """Generate migration appendix document."""
    
    # Group by priority
    by_priority = {
        'critical': [],
        'high': [],
        'medium': [],
        'low': []
    }
    
    for calib in embedded_calibrations:
        by_priority[calib.migration_priority].append(calib)
    
    # Generate markdown report
    lines = [
        "# Embedded Calibration Migration Appendix",
        "",
        "This document tracks all methods with embedded/inline calibration that must be",
        "migrated to the centralized calibration system.",
        "",
        "**Status:** Transitional Anomalies - Explicitly Tracked",
        "",
        f"**Generated:** {datetime.utcnow().isoformat()}",
        "",
        "## Summary",
        "",
        f"- **Total methods with embedded calibration:** {len(embedded_calibrations)}",
    ]
    
    for priority in ['critical', 'high', 'medium', 'low']:
        count = len(by_priority[priority])
        lines.append(f"- **{priority.upper()} priority:** {count}")
    
    lines.extend([
        "",
        "## Migration Backlog",
        "",
        "Methods are listed in priority order (critical → low).",
        ""
    ])
    
    # Detail each priority group
    for priority in ['critical', 'high', 'medium', 'low']:
        if not by_priority[priority]:
            continue
        
        lines.extend([
            f"### {priority.upper()} Priority",
            "",
            f"Count: {len(by_priority[priority])}",
            ""
        ])
        
        for i, calib in enumerate(by_priority[priority], 1):
            lines.extend([
                f"#### {i}. {calib.canonical_name}",
                "",
                f"- **File:** `{calib.file_path}:{calib.line_number}`",
                f"- **Pattern:** {calib.pattern_type}",
                f"- **Complexity:** {calib.migration_complexity}",
                f"- **Parameters found:** {len(calib.parameters_found)}",
                ""
            ])
            
            # Show parameters
            if calib.parameters_found:
                lines.append("**Parameters:**")
                for param in calib.parameters_found[:10]:  # Max 10
                    lines.append(f"- `{param['name']}` = `{param['value']}`")
                lines.append("")
            
            # Notes
            if calib.notes:
                lines.append(f"**Notes:** {calib.notes}")
                lines.append("")
            
            lines.append("---")
            lines.append("")
    
    # Write markdown
    md_path = output_path.with_suffix('.md')
    with open(md_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"Migration appendix written to: {md_path}")
    
    # Also write JSON for machine processing
    json_data = {
        'metadata': {
            'generated_at': datetime.utcnow().isoformat(),
            'total_embedded': len(embedded_calibrations),
            'by_priority': {p: len(by_priority[p]) for p in by_priority}
        },
        'embedded_calibrations': [asdict(c) for c in embedded_calibrations]
    }
    
    with open(output_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"Machine-readable appendix written to: {output_path}")


def main():
    """Main entry point."""
    repo_root = Path(__file__).parent.parent
    catalog_path = repo_root / "config" / "canonical_method_catalog.json"
    output_path = repo_root / "config" / "embedded_calibration_appendix.json"
    
    print("=" * 80)
    print("EMBEDDED CALIBRATION DETECTOR")
    print("=" * 80)
    print()
    
    if not catalog_path.exists():
        print(f"Error: Canonical catalog not found at {catalog_path}")
        print("Run build_canonical_method_catalog.py first.")
        return 1
    
    detector = EmbeddedCalibrationDetector(catalog_path)
    embedded = detector.detect_all()
    
    if embedded:
        generate_migration_appendix(embedded, output_path)
        
        print()
        print("=" * 80)
        print("DETECTION COMPLETE")
        print("=" * 80)
        print(f"\nFound {len(embedded)} methods with embedded calibrations")
        print("\nNext steps:")
        print("  1. Review the migration appendix")
        print("  2. Prioritize critical and high priority items")
        print("  3. Migrate embedded calibrations to centralized system")
    else:
        print("\nNo embedded calibrations detected (all methods use centralized calibration)")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
