# recommendation_engine.py - Rule-Based Recommendation Engine
# coding=utf-8
"""
Recommendation Engine - Multi-Level Rule-Based Recommendations
================================================================

This module implements a rule-based recommendation engine that:
1. Loads and validates recommendation rules from JSON files
2. Evaluates conditions against score data at MICRO, MESO, and MACRO levels
3. Generates actionable recommendations with specific interventions
4. Renders templates with context-specific variable substitution

Supports three levels of recommendations:
- MICRO: Question-level recommendations (PA-DIM combinations)
- MESO: Cluster-level recommendations (CL01-CL04)
- MACRO: Plan-level strategic recommendations

Author: Integration Team
Version: 1.0.0
Python: 3.10+
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
import jsonschema

logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES FOR RECOMMENDATIONS
# ============================================================================

@dataclass
class Recommendation:
    """
    Structured recommendation with full intervention details.
    
    Supports both v1.0 (simple) and v2.0 (enhanced with 7 advanced features):
    1. Template parameterization
    2. Execution logic
    3. Measurable indicators
    4. Unambiguous time horizons
    5. Testable verification
    6. Cost tracking
    7. Authority mapping
    """
    rule_id: str
    level: str  # MICRO, MESO, or MACRO
    problem: str
    intervention: str
    indicator: Dict[str, Any]
    responsible: Dict[str, Any]
    horizon: Dict[str, Any]  # Changed from Dict[str, str] to support enhanced fields
    verification: List[Any]  # Changed from List[str] to support structured verification
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Enhanced fields (v2.0) - optional for backward compatibility
    execution: Optional[Dict[str, Any]] = None
    budget: Optional[Dict[str, Any]] = None
    template_id: Optional[str] = None
    template_params: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = asdict(self)
        # Remove None values for cleaner output
        return {k: v for k, v in result.items() if v is not None}


@dataclass
class RecommendationSet:
    """
    Collection of recommendations with metadata
    """
    level: str
    recommendations: List[Recommendation]
    generated_at: str
    total_rules_evaluated: int
    rules_matched: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'level': self.level,
            'recommendations': [r.to_dict() for r in self.recommendations],
            'generated_at': self.generated_at,
            'total_rules_evaluated': self.total_rules_evaluated,
            'rules_matched': self.rules_matched,
            'metadata': self.metadata
        }


# ============================================================================
# RECOMMENDATION ENGINE
# ============================================================================

class RecommendationEngine:
    """
    Core recommendation engine that evaluates rules and generates recommendations
    """
    
    def __init__(
        self,
        rules_path: str = "config/recommendation_rules.json",
        schema_path: str = "rules/recommendation_rules.schema.json"
    ):
        """
        Initialize recommendation engine
        
        Args:
            rules_path: Path to recommendation rules JSON file
            schema_path: Path to JSON schema for validation
        """
        self.rules_path = Path(rules_path)
        self.schema_path = Path(schema_path)
        self.rules: Dict[str, Any] = {}
        self.schema: Dict[str, Any] = {}
        self.rules_by_level: Dict[str, List[Dict[str, Any]]] = {
            'MICRO': [],
            'MESO': [],
            'MACRO': []
        }
        
        # Load rules and schema
        self._load_schema()
        self._load_rules()
        
        logger.info(
            f"Recommendation engine initialized with "
            f"{len(self.rules_by_level['MICRO'])} MICRO, "
            f"{len(self.rules_by_level['MESO'])} MESO, "
            f"{len(self.rules_by_level['MACRO'])} MACRO rules"
        )
    
    def _load_schema(self):
        """Load JSON schema for rule validation"""
        try:
            with open(self.schema_path, 'r', encoding='utf-8') as f:
                self.schema = json.load(f)
            logger.info(f"Loaded recommendation rules schema from {self.schema_path}")
        except Exception as e:
            logger.error(f"Failed to load schema: {e}")
            raise
    
    def _load_rules(self):
        """Load and validate recommendation rules"""
        try:
            with open(self.rules_path, 'r', encoding='utf-8') as f:
                self.rules = json.load(f)
            
            # Validate against schema
            jsonschema.validate(instance=self.rules, schema=self.schema)
            
            # Organize rules by level
            for rule in self.rules.get('rules', []):
                self._validate_rule(rule)
                level = rule.get('level')
                if level in self.rules_by_level:
                    self.rules_by_level[level].append(rule)
            
            logger.info(f"Loaded and validated {len(self.rules.get('rules', []))} rules from {self.rules_path}")
        except jsonschema.ValidationError as e:
            logger.error(f"Rule validation failed: {e.message}")
            raise
        except Exception as e:
            logger.error(f"Failed to load rules: {e}")
            raise
    
    def reload_rules(self):
        """Reload rules from disk (useful for hot-reloading)"""
        self.rules_by_level = {'MICRO': [], 'MESO': [], 'MACRO': []}
        self._load_rules()
    
    # ========================================================================
    # MICRO LEVEL RECOMMENDATIONS
    # ========================================================================
    
    def generate_micro_recommendations(
        self,
        scores: Dict[str, float],
        context: Optional[Dict[str, Any]] = None
    ) -> RecommendationSet:
        """
        Generate MICRO-level recommendations based on PA-DIM scores
        
        Args:
            scores: Dictionary mapping "PA##-DIM##" to scores (0.0-3.0)
            context: Additional context for template rendering
            
        Returns:
            RecommendationSet with matched recommendations
        """
        recommendations = []
        rules_evaluated = 0
        
        for rule in self.rules_by_level['MICRO']:
            rules_evaluated += 1
            
            # Extract condition
            when = rule.get('when', {})
            pa_id = when.get('pa_id')
            dim_id = when.get('dim_id')
            score_lt = when.get('score_lt')
            
            # Build score key
            score_key = f"{pa_id}-{dim_id}"
            
            # Check if condition matches
            if score_key in scores and scores[score_key] < score_lt:
                # Render template
                template = rule.get('template', {})
                rendered = self._render_micro_template(template, pa_id, dim_id, context)
                
                # Create recommendation with enhanced fields (v2.0) if available
                rec = Recommendation(
                    rule_id=rule.get('rule_id'),
                    level='MICRO',
                    problem=rendered['problem'],
                    intervention=rendered['intervention'],
                    indicator=rendered['indicator'],
                    responsible=rendered['responsible'],
                    horizon=rendered['horizon'],
                    verification=rendered['verification'],
                    metadata={
                        'score_key': score_key,
                        'actual_score': scores[score_key],
                        'threshold': score_lt,
                        'gap': score_lt - scores[score_key]
                    },
                    # Enhanced fields (v2.0)
                    execution=rule.get('execution'),
                    budget=rule.get('budget'),
                    template_id=template.get('template_id'),
                    template_params=template.get('template_params')
                )
                recommendations.append(rec)
        
        return RecommendationSet(
            level='MICRO',
            recommendations=recommendations,
            generated_at=datetime.now(timezone.utc).isoformat(),
            total_rules_evaluated=rules_evaluated,
            rules_matched=len(recommendations)
        )
    
    def _render_micro_template(
        self,
        template: Dict[str, Any],
        pa_id: str,
        dim_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Render MICRO template with variable substitution
        
        Variables supported:
        - {{PAxx}}: Policy area (e.g., PA01)
        - {{DIMxx}}: Dimension (e.g., DIM01)
        - {{Q###}}: Question number (from context)
        """
        ctx = context or {}
        
        # Build substitution map
        substitutions = {
            'PAxx': pa_id,
            'DIMxx': dim_id,
            'Q001': ctx.get('question_id', 'Q001'),  # Default or from context
        }
        
        rendered = {}
        for key, value in template.items():
            if isinstance(value, str):
                rendered[key] = self._substitute_variables(value, substitutions)
            else:
                rendered[key] = value
        
        return rendered
    
    # ========================================================================
    # MESO LEVEL RECOMMENDATIONS
    # ========================================================================
    
    def generate_meso_recommendations(
        self,
        cluster_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> RecommendationSet:
        """
        Generate MESO-level recommendations based on cluster performance
        
        Args:
            cluster_data: Dictionary with cluster metrics:
                {
                    'CL01': {'score': 75.0, 'variance': 0.15, 'weak_pa': 'PA02'},
                    'CL02': {'score': 62.0, 'variance': 0.22, 'weak_pa': 'PA05'},
                    ...
                }
            context: Additional context for template rendering
            
        Returns:
            RecommendationSet with matched recommendations
        """
        recommendations = []
        rules_evaluated = 0
        
        for rule in self.rules_by_level['MESO']:
            rules_evaluated += 1
            
            # Extract condition
            when = rule.get('when', {})
            cluster_id = when.get('cluster_id')
            score_band = when.get('score_band')
            variance_level = when.get('variance_level')
            variance_threshold = when.get('variance_threshold')
            weak_pa_id = when.get('weak_pa_id')
            
            # Get cluster data
            cluster = cluster_data.get(cluster_id, {})
            cluster_score = cluster.get('score', 0)
            cluster_variance = cluster.get('variance', 0)
            cluster_weak_pa = cluster.get('weak_pa')
            
            # Check conditions
            if not self._check_meso_conditions(
                cluster_score, cluster_variance, cluster_weak_pa,
                score_band, variance_level, variance_threshold, weak_pa_id
            ):
                continue
            
            # Render template
            template = rule.get('template', {})
            rendered = self._render_meso_template(template, cluster_id, context)
            
            # Create recommendation with enhanced fields (v2.0) if available
            rec = Recommendation(
                rule_id=rule.get('rule_id'),
                level='MESO',
                problem=rendered['problem'],
                intervention=rendered['intervention'],
                indicator=rendered['indicator'],
                responsible=rendered['responsible'],
                horizon=rendered['horizon'],
                verification=rendered['verification'],
                metadata={
                    'cluster_id': cluster_id,
                    'score': cluster_score,
                    'score_band': score_band,
                    'variance': cluster_variance,
                    'variance_level': variance_level,
                    'weak_pa': cluster_weak_pa
                },
                # Enhanced fields (v2.0)
                execution=rule.get('execution'),
                budget=rule.get('budget'),
                template_id=template.get('template_id'),
                template_params=template.get('template_params')
            )
            recommendations.append(rec)
        
        return RecommendationSet(
            level='MESO',
            recommendations=recommendations,
            generated_at=datetime.now(timezone.utc).isoformat(),
            total_rules_evaluated=rules_evaluated,
            rules_matched=len(recommendations)
        )
    
    def _check_meso_conditions(
        self,
        score: float,
        variance: float,
        weak_pa: Optional[str],
        score_band: str,
        variance_level: str,
        variance_threshold: Optional[float],
        weak_pa_id: Optional[str]
    ) -> bool:
        """Check if MESO conditions are met"""
        # Check score band
        if score_band == 'BAJO' and score >= 55:
            return False
        elif score_band == 'MEDIO' and (score < 55 or score >= 75):
            return False
        elif score_band == 'ALTO' and score < 75:
            return False
        
        # Check variance level
        if variance_level == 'BAJA' and variance >= 0.08:
            return False
        elif variance_level == 'MEDIA' and (variance < 0.08 or variance >= 0.18):
            return False
        elif variance_level == 'ALTA':
            if variance_threshold and variance < variance_threshold / 100:
                return False
            elif not variance_threshold and variance < 0.18:
                return False
        
        # Check weak PA if specified
        if weak_pa_id and weak_pa != weak_pa_id:
            return False
        
        return True
    
    def _render_meso_template(
        self,
        template: Dict[str, Any],
        cluster_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Render MESO template with variable substitution"""
        ctx = context or {}
        
        substitutions = {
            'cluster_id': cluster_id,
        }
        
        rendered = {}
        for key, value in template.items():
            if isinstance(value, str):
                rendered[key] = self._substitute_variables(value, substitutions)
            else:
                rendered[key] = value
        
        return rendered
    
    # ========================================================================
    # MACRO LEVEL RECOMMENDATIONS
    # ========================================================================
    
    def generate_macro_recommendations(
        self,
        macro_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> RecommendationSet:
        """
        Generate MACRO-level strategic recommendations
        
        Args:
            macro_data: Dictionary with plan-level metrics:
                {
                    'macro_band': 'SATISFACTORIO',
                    'clusters_below_target': ['CL02', 'CL03'],
                    'variance_alert': 'MODERADA',
                    'priority_micro_gaps': ['PA01-DIM05', 'PA04-DIM04']
                }
            context: Additional context for template rendering
            
        Returns:
            RecommendationSet with matched recommendations
        """
        recommendations = []
        rules_evaluated = 0
        
        for rule in self.rules_by_level['MACRO']:
            rules_evaluated += 1
            
            # Extract condition
            when = rule.get('when', {})
            macro_band = when.get('macro_band')
            clusters_below = set(when.get('clusters_below_target', []))
            variance_alert = when.get('variance_alert')
            priority_gaps = set(when.get('priority_micro_gaps', []))
            
            # Get macro data
            actual_band = macro_data.get('macro_band')
            actual_clusters = set(macro_data.get('clusters_below_target', []))
            actual_variance = macro_data.get('variance_alert')
            actual_gaps = set(macro_data.get('priority_micro_gaps', []))
            
            # Check conditions
            if macro_band and macro_band != actual_band:
                continue
            if variance_alert and variance_alert != actual_variance:
                continue
            
            # Check if clusters match (subset or exact match)
            if clusters_below and not clusters_below.issubset(actual_clusters):
                # For MACRO, we want exact match or the rule's clusters to be present
                if clusters_below != actual_clusters and not actual_clusters.issubset(clusters_below):
                    continue
            
            # Check if priority gaps match (subset)
            if priority_gaps and not priority_gaps.issubset(actual_gaps):
                continue
            
            # Render template
            template = rule.get('template', {})
            rendered = self._render_macro_template(template, context)
            
            # Create recommendation with enhanced fields (v2.0) if available
            rec = Recommendation(
                rule_id=rule.get('rule_id'),
                level='MACRO',
                problem=rendered['problem'],
                intervention=rendered['intervention'],
                indicator=rendered['indicator'],
                responsible=rendered['responsible'],
                horizon=rendered['horizon'],
                verification=rendered['verification'],
                metadata={
                    'macro_band': actual_band,
                    'clusters_below_target': list(actual_clusters),
                    'variance_alert': actual_variance,
                    'priority_micro_gaps': list(actual_gaps)
                },
                # Enhanced fields (v2.0)
                execution=rule.get('execution'),
                budget=rule.get('budget'),
                template_id=template.get('template_id'),
                template_params=template.get('template_params')
            )
            recommendations.append(rec)
        
        return RecommendationSet(
            level='MACRO',
            recommendations=recommendations,
            generated_at=datetime.now(timezone.utc).isoformat(),
            total_rules_evaluated=rules_evaluated,
            rules_matched=len(recommendations)
        )
    
    def _render_macro_template(
        self,
        template: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Render MACRO template with variable substitution"""
        ctx = context or {}
        
        substitutions = {}
        
        rendered = {}
        for key, value in template.items():
            if isinstance(value, str):
                rendered[key] = self._substitute_variables(value, substitutions)
            else:
                rendered[key] = value
        
        return rendered
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def _substitute_variables(self, text: str, substitutions: Dict[str, str]) -> str:
        """
        Substitute variables in text using {{variable}} syntax
        
        Args:
            text: Text with variables
            substitutions: Dictionary of variable_name -> value
            
        Returns:
            Text with variables substituted
        """
        result = text
        for var, value in substitutions.items():
            pattern = r'\{\{' + re.escape(var) + r'\}\}'
            result = re.sub(pattern, value, result)
        return result

    # ========================================================================
    # VALIDATION UTILITIES
    # ========================================================================

    def _validate_rule(self, rule: Dict[str, Any]) -> None:
        """Apply structural validation to guarantee rigorous recommendations."""
        rule_id = rule.get('rule_id')
        if not isinstance(rule_id, str) or not rule_id.strip():
            raise ValueError("Recommendation rule missing rule_id")

        level = rule.get('level')
        if level not in self.rules_by_level:
            raise ValueError(f"Rule {rule_id} declares unsupported level: {level}")

        when = rule.get('when', {})
        if not isinstance(when, dict):
            raise ValueError(f"Rule {rule_id} has invalid 'when' definition")

        if level == 'MICRO':
            self._validate_micro_when(rule_id, when)
        elif level == 'MESO':
            self._validate_meso_when(rule_id, when)
        elif level == 'MACRO':
            self._validate_macro_when(rule_id, when)

        template = rule.get('template')
        if not isinstance(template, dict):
            raise ValueError(f"Rule {rule_id} lacks a structured template")

        self._validate_template(rule_id, template)

    def _validate_micro_when(self, rule_id: str, when: Dict[str, Any]) -> None:
        required_keys = ('pa_id', 'dim_id', 'score_lt')
        for key in required_keys:
            if key not in when:
                raise ValueError(f"Rule {rule_id} missing '{key}' in MICRO condition")

        pa_id = when['pa_id']
        dim_id = when['dim_id']
        if not isinstance(pa_id, str) or not pa_id.strip():
            raise ValueError(f"Rule {rule_id} has invalid pa_id")
        if not isinstance(dim_id, str) or not dim_id.strip():
            raise ValueError(f"Rule {rule_id} has invalid dim_id")

        score_lt = when['score_lt']
        if not self._is_number(score_lt):
            raise ValueError(f"Rule {rule_id} has non numeric MICRO threshold")
        if not 0 <= float(score_lt) <= 3:
            raise ValueError(f"Rule {rule_id} MICRO threshold must be between 0 and 3")

    def _validate_meso_when(self, rule_id: str, when: Dict[str, Any]) -> None:
        cluster_id = when.get('cluster_id')
        if not isinstance(cluster_id, str) or not cluster_id.strip():
            raise ValueError(f"Rule {rule_id} missing cluster_id for MESO condition")

        condition_counter = 0

        score_band = when.get('score_band')
        if score_band is not None:
            if score_band not in {'BAJO', 'MEDIO', 'ALTO'}:
                raise ValueError(f"Rule {rule_id} has invalid MESO score_band")
            condition_counter += 1

        variance_level = when.get('variance_level')
        if variance_level is not None:
            if variance_level not in {'BAJA', 'MEDIA', 'ALTA'}:
                raise ValueError(f"Rule {rule_id} has invalid MESO variance_level")
            condition_counter += 1

        variance_threshold = when.get('variance_threshold')
        if variance_threshold is not None:
            if not self._is_number(variance_threshold):
                raise ValueError(f"Rule {rule_id} has non-numeric variance_threshold")

        weak_pa_id = when.get('weak_pa_id')
        if weak_pa_id is not None:
            if not isinstance(weak_pa_id, str) or not weak_pa_id.strip():
                raise ValueError(f"Rule {rule_id} has invalid weak_pa_id")
            condition_counter += 1

        if condition_counter == 0:
            raise ValueError(
                f"Rule {rule_id} must specify at least one discriminant condition for MESO"
            )

    def _validate_macro_when(self, rule_id: str, when: Dict[str, Any]) -> None:
        discriminants = 0

        macro_band = when.get('macro_band')
        if macro_band is not None:
            if not isinstance(macro_band, str) or not macro_band.strip():
                raise ValueError(f"Rule {rule_id} has invalid macro_band")
            discriminants += 1

        clusters = when.get('clusters_below_target')
        if clusters is not None:
            if not isinstance(clusters, list) or not clusters:
                raise ValueError(f"Rule {rule_id} must declare non empty clusters_below_target")
            if not all(isinstance(item, str) and item.strip() for item in clusters):
                raise ValueError(f"Rule {rule_id} has invalid cluster identifiers")
            discriminants += 1

        variance_alert = when.get('variance_alert')
        if variance_alert is not None:
            if not isinstance(variance_alert, str) or not variance_alert.strip():
                raise ValueError(f"Rule {rule_id} has invalid variance_alert")
            discriminants += 1

        priority_gaps = when.get('priority_micro_gaps')
        if priority_gaps is not None:
            if not isinstance(priority_gaps, list) or not priority_gaps:
                raise ValueError(f"Rule {rule_id} must declare non-empty priority_micro_gaps")
            if not all(isinstance(item, str) and item.strip() for item in priority_gaps):
                raise ValueError(f"Rule {rule_id} has invalid priority_micro_gaps entries")
            discriminants += 1

        if discriminants == 0:
            raise ValueError(
                f"Rule {rule_id} must specify at least one MACRO discriminant condition"
            )

    def _validate_template(self, rule_id: str, template: Dict[str, Any]) -> None:
        required_fields = ['problem', 'intervention', 'indicator', 'responsible', 'horizon', 'verification']
        for field in required_fields:
            if field not in template:
                raise ValueError(f"Rule {rule_id} template missing '{field}'")

        for text_field in ('problem', 'intervention'):
            value = template[text_field]
            if not isinstance(value, str):
                raise ValueError(f"Rule {rule_id} template field '{text_field}' must be text")
            stripped = value.strip()
            if len(stripped) < 40 or len(stripped.split()) < 12:
                raise ValueError(
                    f"Rule {rule_id} template field '{text_field}' lacks actionable detail"
                )

        indicator = template['indicator']
        if not isinstance(indicator, dict):
            raise ValueError(f"Rule {rule_id} indicator must be an object")
        for key in ('name', 'target', 'unit'):
            if key not in indicator:
                raise ValueError(f"Rule {rule_id} indicator missing '{key}' field")

        if not isinstance(indicator['name'], str) or len(indicator['name'].strip()) < 5:
            raise ValueError(f"Rule {rule_id} indicator name too short")

        target = indicator['target']
        if not self._is_number(target):
            raise ValueError(f"Rule {rule_id} indicator target must be numeric")

        unit = indicator['unit']
        if not isinstance(unit, str) or not unit.strip():
            raise ValueError(f"Rule {rule_id} indicator unit missing or empty")

        acceptable_range = indicator.get('acceptable_range')
        if acceptable_range is not None:
            if not isinstance(acceptable_range, list) or len(acceptable_range) != 2:
                raise ValueError(f"Rule {rule_id} acceptable_range must have two numeric bounds")
            if not all(self._is_number(bound) for bound in acceptable_range):
                raise ValueError(f"Rule {rule_id} acceptable_range values must be numeric")
            lower, upper = acceptable_range
            if float(lower) >= float(upper):
                raise ValueError(f"Rule {rule_id} acceptable_range lower bound must be < upper bound")

        responsible = template['responsible']
        if not isinstance(responsible, dict):
            raise ValueError(f"Rule {rule_id} responsible must be an object")
        for key in ('entity', 'role'):
            value = responsible.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Rule {rule_id} responsible missing '{key}'")

        partners = responsible.get('partners')
        if partners is None or not isinstance(partners, list) or not partners:
            raise ValueError(f"Rule {rule_id} responsible must enumerate partners")
        if any(not isinstance(partner, str) or not partner.strip() for partner in partners):
            raise ValueError(f"Rule {rule_id} responsible partners must be non empty strings")

        horizon = template['horizon']
        if not isinstance(horizon, dict):
            raise ValueError(f"Rule {rule_id} horizon must be an object")
        for key in ('start', 'end'):
            value = horizon.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Rule {rule_id} horizon missing '{key}'")

        verification = template['verification']
        if not isinstance(verification, list) or not verification:
            raise ValueError(f"Rule {rule_id} must define verification artifacts")
        for artifact in verification:
            if isinstance(artifact, str):
                if len(artifact.strip()) < 10:
                    raise ValueError(
                        f"Rule {rule_id} verification entries must describe concrete artifacts"
                    )
            elif isinstance(artifact, dict):
                if not artifact.get('id') or not artifact.get('type'):
                    raise ValueError(
                        f"Rule {rule_id} structured verification entries require 'id' and 'type'"
                    )
            else:
                raise ValueError(
                    f"Rule {rule_id} verification entries must be strings or structured dictionaries"
                )

    @staticmethod
    def _is_number(value: Any) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    
    def generate_all_recommendations(
        self,
        micro_scores: Dict[str, float],
        cluster_data: Dict[str, Any],
        macro_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, RecommendationSet]:
        """
        Generate recommendations at all three levels
        
        Args:
            micro_scores: PA-DIM scores for MICRO recommendations
            cluster_data: Cluster metrics for MESO recommendations
            macro_data: Plan-level metrics for MACRO recommendations
            context: Additional context
            
        Returns:
            Dictionary with 'MICRO', 'MESO', and 'MACRO' recommendation sets
        """
        return {
            'MICRO': self.generate_micro_recommendations(micro_scores, context),
            'MESO': self.generate_meso_recommendations(cluster_data, context),
            'MACRO': self.generate_macro_recommendations(macro_data, context)
        }
    
    def export_recommendations(
        self,
        recommendations: Dict[str, RecommendationSet],
        output_path: str,
        format: str = 'json'
    ):
        """
        Export recommendations to file
        
        Args:
            recommendations: Dictionary of recommendation sets
            output_path: Path to output file
            format: Output format ('json' or 'markdown')
        """
        if format == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(
                    {level: rec_set.to_dict() for level, rec_set in recommendations.items()},
                    f,
                    indent=2,
                    ensure_ascii=False
                )
        elif format == 'markdown':
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(self._format_as_markdown(recommendations))
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Exported recommendations to {output_path} in {format} format")
    
    def _format_as_markdown(self, recommendations: Dict[str, RecommendationSet]) -> str:
        """Format recommendations as Markdown"""
        lines = ["# Recomendaciones del Plan de Desarrollo\n"]
        
        for level in ['MICRO', 'MESO', 'MACRO']:
            rec_set = recommendations.get(level)
            if not rec_set:
                continue
            
            lines.append(f"\n## Nivel {level}\n")
            lines.append(f"**Generado:** {rec_set.generated_at}\n")
            lines.append(f"**Reglas evaluadas:** {rec_set.total_rules_evaluated}\n")
            lines.append(f"**Recomendaciones:** {rec_set.rules_matched}\n")
            
            for i, rec in enumerate(rec_set.recommendations, 1):
                lines.append(f"\n### {i}. {rec.rule_id}\n")
                lines.append(f"**Problema:** {rec.problem}\n")
                lines.append(f"\n**Intervención:** {rec.intervention}\n")
                lines.append(f"\n**Indicador:**")
                lines.append(f"- Nombre: {rec.indicator.get('name')}")
                lines.append(f"- Meta: {rec.indicator.get('target')} {rec.indicator.get('unit')}\n")
                lines.append(f"\n**Responsable:** {rec.responsible.get('entity')} ({rec.responsible.get('role')})\n")
                lines.append(f"**Socios:** {', '.join(rec.responsible.get('partners', []))}\n")
                lines.append(f"\n**Horizonte:** {rec.horizon.get('start')} → {rec.horizon.get('end')}\n")
                lines.append(f"\n**Verificación:**")
                for v in rec.verification:
                    lines.append(f"- {v}")
                lines.append("")
        
        return "\n".join(lines)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def load_recommendation_engine(
    rules_path: str = "config/recommendation_rules.json",
    schema_path: str = "rules/recommendation_rules.schema.json"
) -> RecommendationEngine:
    """
    Convenience function to load recommendation engine
    
    Args:
        rules_path: Path to rules JSON
        schema_path: Path to schema JSON
        
    Returns:
        Initialized RecommendationEngine
    """
    return RecommendationEngine(rules_path=rules_path, schema_path=schema_path)


if __name__ == '__main__':
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Initialize engine
    engine = load_recommendation_engine()
    
    # Example MICRO recommendations
    micro_scores = {
        'PA01-DIM01': 1.2,  # Below threshold of 1.65
        'PA02-DIM03': 1.8,  # Above threshold
        'PA03-DIM05': 1.4,  # Below threshold
    }
    
    micro_recs = engine.generate_micro_recommendations(micro_scores)
    print(f"\n=== MICRO Recommendations ===")
    print(f"Rules evaluated: {micro_recs.total_rules_evaluated}")
    print(f"Recommendations: {micro_recs.rules_matched}")
    
    for rec in micro_recs.recommendations[:2]:  # Show first 2
        print(f"\n{rec.rule_id}:")
        print(f"Problem: {rec.problem[:100]}...")
        print(f"Intervention: {rec.intervention[:100]}...")
    
    # Example MESO recommendations
    cluster_data = {
        'CL01': {'score': 72.0, 'variance': 0.25, 'weak_pa': 'PA02'},
        'CL02': {'score': 58.0, 'variance': 0.12},
    }
    
    meso_recs = engine.generate_meso_recommendations(cluster_data)
    print(f"\n=== MESO Recommendations ===")
    print(f"Rules evaluated: {meso_recs.total_rules_evaluated}")
    print(f"Recommendations: {meso_recs.rules_matched}")
    
    # Example MACRO recommendations
    macro_data = {
        'macro_band': 'SATISFACTORIO',
        'clusters_below_target': ['CL02', 'CL03'],
        'variance_alert': 'MODERADA',
        'priority_micro_gaps': ['PA01-DIM05', 'PA05-DIM04', 'PA04-DIM04', 'PA08-DIM05']
    }
    
    macro_recs = engine.generate_macro_recommendations(macro_data)
    print(f"\n=== MACRO Recommendations ===")
    print(f"Rules evaluated: {macro_recs.total_rules_evaluated}")
    print(f"Recommendations: {macro_recs.rules_matched}")
