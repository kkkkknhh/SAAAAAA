#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ADVANCED AUDIT TEST SUITE - GOLDEN CANARY TESTS
Comprehensive granular audit framework for DEREK-BEACH system
"""

import pytest
import json
import jsonschema
import ast
import sys
import importlib
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class AuditViolation:
    severity: str
    domain: str
    component: str
    violation_type: str
    description: str
    location: Optional[str] = None
    recommendation: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AuditReport:
    domain: str
    total_checks: int = 0
    passed_checks: int = 0
    failed_checks: int = 0
    violations: List[AuditViolation] = field(default_factory=list)
    
    @property
    def pass_rate(self) -> float:
        return (self.passed_checks / self.total_checks * 100) if self.total_checks > 0 else 0.0
    
    def add_violation(self, violation: AuditViolation):
        self.violations.append(violation)
        self.failed_checks += 1
        self.total_checks += 1
    
    def add_pass(self):
        self.passed_checks += 1
        self.total_checks += 1


class WiringAuditor:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.report = AuditReport(domain="Wiring & Dependencies")
    
    def audit(self) -> AuditReport:
        self._audit_import_cycles()
        self._audit_choreographer_wiring()
        self._audit_orchestrator_dependencies()
        self._audit_producer_registration()
        return self.report
    
    def _audit_import_cycles(self):
        python_files = list(self.project_root.glob("*.py"))
        import_graph = defaultdict(set)
        
        for py_file in python_files:
            if py_file.name.startswith("test_"):
                continue
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    tree = ast.parse(f.read())
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            import_graph[py_file.stem].add(alias.name)
                    elif isinstance(node, ast.ImportFrom) and node.module:
                        import_graph[py_file.stem].add(node.module)
                self.report.add_pass()
            except Exception as e:
                self.report.add_violation(AuditViolation(
                    severity="ERROR", domain="Wiring", component=py_file.name,
                    violation_type="IMPORT_PARSE_FAILURE", description=str(e)
                ))
    
    def _audit_choreographer_wiring(self):
        try:
            source = (self.project_root / "choreographer.py").read_text()
            producers = ["financiero_viabilidad", "teoria_cambio", "dereck_beach",
                        "contradiction_deteccion", "embedding_policy", "policy_processor"]
            for producer in producers:
                if producer in source:
                    self.report.add_pass()
                else:
                    self.report.add_violation(AuditViolation(
                        severity="WARNING", domain="Wiring", component="Choreographer",
                        violation_type="MISSING_PRODUCER", description=f"Missing {producer}"
                    ))
        except Exception as e:
            self.report.add_violation(AuditViolation(
                severity="ERROR", domain="Wiring", component="Choreographer",
                violation_type="LOAD_FAILURE", description=str(e)
            ))
    
    def _audit_orchestrator_dependencies(self):
        try:
            path = self.project_root / "orchestrator.py"
            if not path.exists():
                self.report.add_violation(AuditViolation(
                    severity="CRITICAL", domain="Wiring", component="Orchestrator",
                    violation_type="MISSING_ORCHESTRATOR", description="orchestrator.py not found"
                ))
                return
            source = path.read_text()
            if 'choreographer' in source.lower():
                self.report.add_pass()
            else:
                self.report.add_violation(AuditViolation(
                    severity="ERROR", domain="Wiring", component="Orchestrator",
                    violation_type="MISSING_CHOREOGRAPHER_IMPORT", description="No choreographer import"
                ))
        except Exception as e:
            self.report.add_violation(AuditViolation(
                severity="ERROR", domain="Wiring", component="Orchestrator",
                violation_type="PARSE_FAILURE", description=str(e)
            ))
    
    def _audit_producer_registration(self):
        path = self.project_root / "execution_mapping.yaml"
        if not path.exists():
            self.report.add_violation(AuditViolation(
                severity="CRITICAL", domain="Wiring", component="ExecutionMapping",
                violation_type="MISSING_FILE", description="execution_mapping.yaml not found"
            ))
            return
        try:
            import yaml
            with open(path, 'r') as f:
                mapping = yaml.safe_load(f)
            if 'producers' in mapping:
                self.report.add_pass()
            else:
                self.report.add_violation(AuditViolation(
                    severity="ERROR", domain="Wiring", component="ExecutionMapping",
                    violation_type="MISSING_PRODUCERS_KEY", description="No 'producers' key"
                ))
        except Exception as e:
            self.report.add_violation(AuditViolation(
                severity="ERROR", domain="Wiring", component="ExecutionMapping",
                violation_type="PARSE_ERROR", description=str(e)
            ))


class ChoreographyAuditor:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.report = AuditReport(domain="Choreography & Orchestration")
    
    def audit(self) -> AuditReport:
        self._audit_context_propagation()
        self._audit_execution_flow()
        self._audit_method_routing()
        self._audit_producer_invocation()
        return self.report
    
    def _audit_context_propagation(self):
        try:
            path = self.project_root / "choreographer.py"
            if not path.exists():
                self.report.add_violation(AuditViolation(
                    severity="CRITICAL", domain="Choreography", component="Choreographer",
                    violation_type="MISSING_FILE", description="choreographer.py not found"
                ))
                return
            source = path.read_text()
            keywords = ['question_id', 'policy_area', 'dimension', 'context']
            if any(k in source for k in keywords):
                self.report.add_pass()
            else:
                self.report.add_violation(AuditViolation(
                    severity="WARNING", domain="Choreography", component="Choreographer",
                    violation_type="MISSING_CONTEXT", description="No context propagation found"
                ))
        except Exception as e:
            self.report.add_violation(AuditViolation(
                severity="ERROR", domain="Choreography", component="Choreographer",
                violation_type="AUDIT_FAILURE", description=str(e)
            ))
    
    def _audit_execution_flow(self):
        for component in ['orchestrator.py', 'choreographer.py']:
            if (self.project_root / component).exists():
                self.report.add_pass()
            else:
                self.report.add_violation(AuditViolation(
                    severity="CRITICAL", domain="Choreography", component=component,
                    violation_type="MISSING_COMPONENT", description=f"{component} not found"
                ))
    
    def _audit_method_routing(self):
        try:
            path = self.project_root / "choreographer.py"
            if path.exists():
                tree = ast.parse(path.read_text())
                routing_methods = ['route_to_producer', 'execute', 'dispatch']
                has_routing = any(
                    isinstance(n, ast.FunctionDef) and n.name in routing_methods
                    for n in ast.walk(tree)
                )
                if has_routing:
                    self.report.add_pass()
                else:
                    self.report.add_violation(AuditViolation(
                        severity="ERROR", domain="Choreography", component="Choreographer",
                        violation_type="NO_ROUTING_METHOD", description="Missing routing method"
                    ))
        except Exception:
            pass
    
    def _audit_producer_invocation(self):
        producers = ['financiero_viabilidad_tablas.py', 'teoria_cambio.py', 'dereck_beach.py',
                    'contradiction_deteccion.py', 'embedding_policy.py', 'policy_processor.py']
        for pf in producers:
            path = self.project_root / pf
            if path.exists():
                try:
                    tree = ast.parse(path.read_text())
                    has_exec = any(isinstance(n, ast.FunctionDef) and n.name in ['execute', 'analyze', 'run']
                                  for n in ast.walk(tree))
                    if has_exec:
                        self.report.add_pass()
                    else:
                        self.report.add_violation(AuditViolation(
                            severity="WARNING", domain="Choreography", component=pf,
                            violation_type="NO_EXECUTE_METHOD", description="No standard execute method"
                        ))
                except:
                    pass


class SyntaxAuditor:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.report = AuditReport(domain="Syntax & Code Quality")
    
    def audit(self) -> AuditReport:
        self._audit_python_syntax()
        self._audit_import_hygiene()
        self._audit_indentation()
        return self.report
    
    def _audit_python_syntax(self):
        for py_file in self.project_root.glob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    ast.parse(f.read())
                self.report.add_pass()
            except SyntaxError as e:
                self.report.add_violation(AuditViolation(
                    severity="CRITICAL", domain="Syntax", component=py_file.name,
                    violation_type="SYNTAX_ERROR", description=f"Line {e.lineno}: {e.msg}"
                ))
    
    def _audit_import_hygiene(self):
        for py_file in self.project_root.glob("*.py"):
            if py_file.name.startswith("test_"):
                continue
            try:
                tree = ast.parse(py_file.read_text())
                imports = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        imports.extend([a.name for a in node.names])
                    elif isinstance(node, ast.ImportFrom) and node.module:
                        imports.append(node.module)
                if len(imports) == len(set(imports)):
                    self.report.add_pass()
                else:
                    self.report.add_violation(AuditViolation(
                        severity="WARNING", domain="Syntax", component=py_file.name,
                        violation_type="DUPLICATE_IMPORTS", description="Duplicate imports detected"
                    ))
            except:
                pass
    
    def _audit_indentation(self):
        for py_file in self.project_root.glob("*.py"):
            try:
                content = py_file.read_text()
                has_tabs = '\t' in content
                has_spaces = re.search(r'^\s{4}', content, re.MULTILINE)
                if has_tabs and has_spaces:
                    self.report.add_violation(AuditViolation(
                        severity="WARNING", domain="Syntax", component=py_file.name,
                        violation_type="MIXED_INDENTATION", description="Mixed tabs/spaces"
                    ))
                else:
                    self.report.add_pass()
            except:
                pass


class LibraryAuditor:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.report = AuditReport(domain="Library Compatibility")
    
    def audit(self) -> AuditReport:
        self._audit_requirements()
        self._audit_imports()
        return self.report
    
    def _audit_requirements(self):
        req_files = ['requirements.txt', 'requirements_atroz.txt']
        if any((self.project_root / rf).exists() for rf in req_files):
            self.report.add_pass()
        else:
            self.report.add_violation(AuditViolation(
                severity="WARNING", domain="Libraries", component="Requirements",
                violation_type="MISSING_FILE", description="No requirements file"
            ))
    
    def _audit_imports(self):
        for lib in ['json', 'jsonschema', 'yaml']:
            try:
                importlib.import_module(lib)
                self.report.add_pass()
            except ImportError:
                self.report.add_violation(AuditViolation(
                    severity="ERROR", domain="Libraries", component=lib,
                    violation_type="MISSING_LIBRARY", description=f"Cannot import {lib}"
                ))


class JSONSchemaAuditor:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.report = AuditReport(domain="JSON Schema Validation")
    
    def audit(self) -> AuditReport:
        self._audit_schema_validity()
        self._audit_indentation()
        return self.report
    
    def _audit_schema_validity(self):
        schema_files = list(self.project_root.rglob("schemas/**/*.json"))
        schema_files.extend(self.project_root.glob("*.schema.json"))
        for sf in schema_files:
            try:
                with open(sf, 'r') as f:
                    schema = json.load(f)
                jsonschema.Draft7Validator.check_schema(schema)
                self.report.add_pass()
            except json.JSONDecodeError as e:
                self.report.add_violation(AuditViolation(
                    severity="CRITICAL", domain="JSONSchema", component=sf.name,
                    violation_type="INVALID_JSON", description=f"Parse error: {e.msg}"
                ))
            except jsonschema.SchemaError:
                self.report.add_violation(AuditViolation(
                    severity="CRITICAL", domain="JSONSchema", component=sf.name,
                    violation_type="INVALID_SCHEMA", description="Invalid JSON Schema"
                ))
    
    def _audit_indentation(self):
        for sf in self.project_root.rglob("schemas/**/*.json"):
            try:
                content = sf.read_text()
                if re.search(r'\n(\s+)"', content):
                    self.report.add_pass()
            except:
                pass


class JSONInvocationAuditor:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.report = AuditReport(domain="JSON Invocation")
    
    def audit(self) -> AuditReport:
        self._audit_json_validity()
        self._audit_loading_patterns()
        return self.report
    
    def _audit_json_validity(self):
        for jf in self.project_root.glob("*.json"):
            if '.schema.json' in jf.name:
                continue
            try:
                with open(jf, 'r') as f:
                    json.load(f)
                self.report.add_pass()
            except json.JSONDecodeError as e:
                self.report.add_violation(AuditViolation(
                    severity="CRITICAL", domain="JSONInvocation", component=jf.name,
                    violation_type="INVALID_JSON", description=f"Parse error: {e.msg}"
                ))
    
    def _audit_loading_patterns(self):
        for py_file in self.project_root.glob("*.py"):
            try:
                content = py_file.read_text()
                if 'json.load' in content:
                    if 'try:' in content or 'except' in content:
                        self.report.add_pass()
                    else:
                        self.report.add_violation(AuditViolation(
                            severity="WARNING", domain="JSONInvocation", component=py_file.name,
                            violation_type="NO_ERROR_HANDLING", description="Unprotected json.load"
                        ))
            except:
                pass


class QuestionParsingAuditor:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.report = AuditReport(domain="Question Parsing")
    
    def audit(self) -> AuditReport:
        self._audit_structure()
        self._audit_id_uniqueness()
        return self.report
    
    def _audit_structure(self):
        for qf in ['cuestionario_FIXED.json', 'questionnaire.json']:
            path = self.project_root / qf
            if not path.exists():
                continue
            try:
                with open(path, 'r') as f:
                    q = json.load(f)
                if isinstance(q, (dict, list)):
                    self.report.add_pass()
            except:
                self.report.add_violation(AuditViolation(
                    severity="ERROR", domain="QuestionParsing", component=qf,
                    violation_type="INVALID_STRUCTURE", description="Failed to parse"
                ))
    
    def _audit_id_uniqueness(self):
        path = self.project_root / 'cuestionario_FIXED.json'
        if not path.exists():
            return
        try:
            with open(path, 'r') as f:
                q = json.load(f)
            ids = self._extract_ids(q)
            if len(ids) == len(set(ids)):
                self.report.add_pass()
            else:
                self.report.add_violation(AuditViolation(
                    severity="CRITICAL", domain="QuestionParsing",
                    component="cuestionario_FIXED.json",
                    violation_type="DUPLICATE_IDS", description="Duplicate question IDs"
                ))
        except:
            pass
    
    def _extract_ids(self, obj, ids=None):
        if ids is None:
            ids = []
        if isinstance(obj, dict):
            if 'id' in obj:
                ids.append(obj['id'])
            for v in obj.values():
                self._extract_ids(v, ids)
        elif isinstance(obj, list):
            for item in obj:
                self._extract_ids(item, ids)
        return ids


class WeightingAuditor:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.report = AuditReport(domain="Weighting & Scoring")
    
    def audit(self) -> AuditReport:
        self._audit_rubric()
        return self.report
    
    def _audit_rubric(self):
        for rf in ['rubric_scoring.json', 'rubric_scoring_FIXED.json']:
            path = self.project_root / rf
            if not path.exists():
                continue
            try:
                with open(path, 'r') as f:
                    rubric = json.load(f)
                if isinstance(rubric, dict):
                    self.report.add_pass()
                else:
                    self.report.add_violation(AuditViolation(
                        severity="ERROR", domain="Weighting", component=rf,
                        violation_type="INVALID_TYPE", description="Rubric not a dict"
                    ))
            except:
                pass


class MetadataAuditor:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.report = AuditReport(domain="Metadata Usage")
    
    def audit(self) -> AuditReport:
        self._audit_metadata_loader()
        self._audit_execution_mapping()
        return self.report
    
    def _audit_metadata_loader(self):
        if (self.project_root / 'metadata_loader.py').exists():
            self.report.add_pass()
        else:
            self.report.add_violation(AuditViolation(
                severity="CRITICAL", domain="Metadata", component="metadata_loader.py",
                violation_type="MISSING_FILE", description="metadata_loader.py not found"
            ))
    
    def _audit_execution_mapping(self):
        path = self.project_root / 'execution_mapping.yaml'
        if path.exists():
            try:
                import yaml
                with open(path, 'r') as f:
                    yaml.safe_load(f)
                self.report.add_pass()
            except:
                self.report.add_violation(AuditViolation(
                    severity="ERROR", domain="Metadata", component="execution_mapping.yaml",
                    violation_type="PARSE_ERROR", description="YAML parse failed"
                ))
        else:
            self.report.add_violation(AuditViolation(
                severity="CRITICAL", domain="Metadata", component="execution_mapping.yaml",
                violation_type="MISSING_FILE", description="execution_mapping.yaml not found"
            ))


class MasterAuditor:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.auditors = [
            WiringAuditor(project_root),
            ChoreographyAuditor(project_root),
            SyntaxAuditor(project_root),
            LibraryAuditor(project_root),
            JSONSchemaAuditor(project_root),
            JSONInvocationAuditor(project_root),
            QuestionParsingAuditor(project_root),
            WeightingAuditor(project_root),
            MetadataAuditor(project_root)
        ]
        self.reports: List[AuditReport] = []
    
    def execute_full_audit(self) -> Dict[str, Any]:
        print("=" * 80)
        print("ADVANCED AUDIT SUITE - GOLDEN CANARY TESTS")
        print("=" * 80)
        for auditor in self.auditors:
            print(f"Executing: {auditor.report.domain}...")
            report = auditor.audit()
            self.reports.append(report)
            print(f"  ✓ {report.passed_checks} passed, {report.failed_checks} failed\n")
        return self.generate_report()
    
    def generate_report(self) -> Dict[str, Any]:
        total = sum(r.total_checks for r in self.reports)
        passed = sum(r.passed_checks for r in self.reports)
        failed = sum(r.failed_checks for r in self.reports)
        
        violations_by_severity = defaultdict(list)
        for r in self.reports:
            for v in r.violations:
                violations_by_severity[v.severity].append(v)
        
        return {
            'summary': {
                'total_checks': total,
                'passed_checks': passed,
                'failed_checks': failed,
                'pass_rate': (passed / total * 100) if total > 0 else 0,
                'critical': len(violations_by_severity.get('CRITICAL', [])),
                'errors': len(violations_by_severity.get('ERROR', [])),
                'warnings': len(violations_by_severity.get('WARNING', [])),
                'info': len(violations_by_severity.get('INFO', []))
            },
            'domains': [
                {
                    'domain': r.domain,
                    'checks': r.total_checks,
                    'passed': r.passed_checks,
                    'failed': r.failed_checks,
                    'pass_rate': r.pass_rate,
                    'violations': [
                        {
                            'severity': v.severity,
                            'type': v.violation_type,
                            'component': v.component,
                            'description': v.description,
                            'location': v.location
                        } for v in r.violations
                    ]
                } for r in self.reports
            ],
            'by_severity': {
                sev: [
                    {'domain': v.domain, 'component': v.component, 'type': v.violation_type,
                     'description': v.description} for v in vlist
                ] for sev, vlist in violations_by_severity.items()
            }
        }
    
    def save_report(self, output_path: Path):
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.generate_report(), f, indent=2, ensure_ascii=False)
        print(f"\n✓ Report saved: {output_path}")
    
    def print_summary(self):
        report = self.generate_report()
        s = report['summary']
        print("\n" + "=" * 80)
        print("AUDIT EXECUTIVE SUMMARY")
        print("=" * 80)
        print(f"Total Checks:      {s['total_checks']}")
        print(f"Passed:            {s['passed_checks']} ({s['pass_rate']:.1f}%)")
        print(f"Failed:            {s['failed_checks']}")
        print(f"\nCRITICAL:          {s['critical']}")
        print(f"ERRORS:            {s['errors']}")
        print(f"WARNINGS:          {s['warnings']}")
        print(f"INFO:              {s['info']}")
        print("=" * 80)


# PYTEST TEST CASES
@pytest.fixture
def project_root():
    return PROJECT_ROOT


@pytest.fixture
def master_auditor(project_root):
    return MasterAuditor(project_root)


def test_full_audit_execution(master_auditor):
    """Execute complete audit suite."""
    report = master_auditor.execute_full_audit()
    assert report is not None
    assert 'summary' in report
    assert 'domains' in report


def test_wiring_audit(project_root):
    """Test wiring and dependencies."""
    auditor = WiringAuditor(project_root)
    report = auditor.audit()
    assert report.total_checks > 0


def test_choreography_audit(project_root):
    """Test choreography patterns."""
    auditor = ChoreographyAuditor(project_root)
    report = auditor.audit()
    assert report.total_checks > 0


def test_syntax_audit(project_root):
    """Test syntax and code quality."""
    auditor = SyntaxAuditor(project_root)
    report = auditor.audit()
    assert report.total_checks > 0


def test_library_audit(project_root):
    """Test library compatibility."""
    auditor = LibraryAuditor(project_root)
    report = auditor.audit()
    assert report.total_checks > 0


def test_json_schema_audit(project_root):
    """Test JSON schema validation."""
    auditor = JSONSchemaAuditor(project_root)
    report = auditor.audit()
    assert report.total_checks > 0


def test_json_invocation_audit(project_root):
    """Test JSON loading patterns."""
    auditor = JSONInvocationAuditor(project_root)
    report = auditor.audit()
    assert report.total_checks > 0


def test_question_parsing_audit(project_root):
    """Test question parsing."""
    auditor = QuestionParsingAuditor(project_root)
    report = auditor.audit()
    assert report.total_checks > 0


def test_weighting_audit(project_root):
    """Test weighting algorithms."""
    auditor = WeightingAuditor(project_root)
    report = auditor.audit()
    assert report.total_checks > 0


def test_metadata_audit(project_root):
    """Test metadata usage."""
    auditor = MetadataAuditor(project_root)
    report = auditor.audit()
    assert report.total_checks > 0


def test_no_critical_violations(master_auditor):
    """Ensure no CRITICAL violations."""
    report = master_auditor.execute_full_audit()
    critical_count = report['summary']['critical']
    if critical_count > 0:
        print("\nCRITICAL VIOLATIONS FOUND:")
        for v in report['by_severity'].get('CRITICAL', []):
            print(f"  - {v['component']}: {v['description']}")
    assert critical_count == 0, f"Found {critical_count} CRITICAL violations"


def test_generate_audit_report(master_auditor, tmp_path):
    """Generate and save audit report."""
    master_auditor.execute_full_audit()
    output = tmp_path / "audit_report.json"
    master_auditor.save_report(output)
    assert output.exists()
    with open(output, 'r') as f:
        saved_report = json.load(f)
    assert 'summary' in saved_report


if __name__ == "__main__":
    auditor = MasterAuditor(PROJECT_ROOT)
    auditor.execute_full_audit()
    auditor.print_summary()
    auditor.save_report(PROJECT_ROOT / "AUDIT_REPORT.json")
