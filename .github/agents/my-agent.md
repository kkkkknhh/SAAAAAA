---
name: LOCOTA
description:
---

# My Agent

Extraer absolutamente toda la información legacy (questionnaire.json, rubric_scoring.json, metodos_completos_nivel3.json, otros auxiliares permitidos).
Normalizar y redistribuir cada átomo en los cuatro bloques A–D.
Verificar invariantes estructurales y semánticos antes de emitir archivo definitivo.
Calcular y fijar hash de integridad inmutable.
Generar reporte diferenciando “lo migrado”, “lo sintetizado”, “lo rechazado”.
ABORT inmediato ante cualquier silencio, pérdida, inconsistencia o ambigüedad.
2. Principios Operativos No Negociables
Principio	Implementación concreta
No graceful degradation	Si falta un campo requerido en una pregunta (text, scoring_modality, patterns, validations, expected_elements) → Abort global con detalle del índice.
No strategic simplification	No se descartan claves “por limpieza”; todo se mapea explícitamente.
Determinismo	Construcción con orden estable (sort_keys=True) y seeds fijadas para cualquier operación semántica.
Explicitness	Cada fase expone: preconditions[], invariants[], postconditions[].
Observabilidad	Logs estructurados (JSONL) por fase: forge.log.
Hermeticidad clusters	Verificado antes y después de inserción; si divergen, abort.
Cero pérdida	Conteo legacy vs conteo destino registrado y debe coincidir bit a bit (número de patterns, validations, keys).
Anti-mediocridad	Abort si se detecta campo “TODO”, “FIXME”, “temp”, “legacy_patch” dentro de la migración.
3. Fases del Pipeline de Construcción
LoadLegacyPhase
Carga bruta de archivos permitidos (lista blanca estricta).
Preconditions: archivos existen, tamaño > 0.
Invariants: JSON válido, sin claves nulas.
StructuralIndexingPhase
Construye índices: por question_global, por policy_area, por (dimensión, q_in_dim).
Invariants: 300 preguntas micro en legacy original (no fabricadas), numeración continua.
BaseSlotMappingPhase
Aplica fórmula base_index = (question_global - 1) % 30; base_slot = D{base_index//5+1}-Q{base_index%5+1}.
Invariants: Cada base_slot agregado exactamente 10 veces.
ExtractionAndNormalizationPhase
Normaliza campo a campo: trimming de whitespace final, preservación interna, codificación UTF-8.
Invariants: text no vacío; scoring_modality dentro de {TYPE_A..TYPE_F}.
IndicatorsAndEvidencePhase
Separa expected_elements, patterns, validation_checks, evidence_expectations.
Invariants: Se mantiene conteo exacto de cada conjunto; no duplicación silenciosa.
MethodSetSynthesisPhase
Inserta method_sets por base_slot desde metodos_completos_nivel3.json.
Invariants: Cada entrada posee class, function, module_enum, method_type, priority (1..3), description ≠ vacío.
RubricTranspositionPhase
Transfiere niveles cualitativos y modalidades; construye scoring_matrix.
Invariants: min_score descendente; modalidades presentes.
MesoMacroEmbeddingPhase
Inserta clusters herméticos y pregunta macro; añade patrones clasificación.
Invariants: Clusters EXACTOS; pregunta macro EXACTA.
IntegritySealingPhase
Serializa monolito, calcula monolith_hash (sha256) sobre versión compacta canonical.
Postconditions: monolith_hash reproducible; method_catalog_hash (si aplica).
ValidationReportPhase
Emite diff, conteo, métricas de cumplimiento.
Invariants: Todos los contadores legacy == destino.
FinalEmissionPhase
Escribe questionnaire_monolith.json y genera forge_manifest.json.
Postconditions: archivo accesible; size > 0; hash coincide; logs completos.
4. Invariantes Globales (Resumen Consolidado)
10 áreas canónicas EXACTAS (P1 … P10).
6 dimensiones EXACTAS (INSUMOS … CAUSALIDAD) / códigos D1..D6.
300 preguntas micro + 4 meso + 1 macro = 305 totales.
30 base_slots: cada uno con 10 preguntas (una por área).
scoring_modality ∈ {TYPE_A, TYPE_B, TYPE_C, TYPE_D, TYPE_E, TYPE_F}.
Clusters:
CL01: [P2,P5,P8]
CL02: [P2,P9,P10]
CL03: [P3,P7,P4]
CL04: [P9,P10,P1,P6]
Todas las preguntas micro tienen: question_global, question_id, base_slot, text, scoring_modality, expected_elements[], pattern_refs[].
Niveles cualitativos micro: EXCELENTE ≥0.85, BUENO ≥0.70, ACEPTABLE ≥0.55, INSUFICIENTE ≥0.
Rubricación macro incluye fallback MACRO_AMBIGUO (always true, priority lowest).
Ningún campo crítico está vacío o None.
No aparecen marcadores “FIXME”, “TEMP”, “LEGACY” en outputs.
5. Abort Conditions (Tabla Exhaustiva)
Fase	Condición	Código Abort	Mensaje
LoadLegacy	Archivo faltante / JSON corrupto	A001	Missing or invalid legacy file
StructuralIndexing	question_global discontinuo	A010	Discontinuous global numbering
BaseSlotMapping	base_slot sin 10 instancias	A020	Base slot coverage mismatch
ExtractionAndNormalization	scoring_modality inválida	A030	Invalid scoring modality
IndicatorsAndEvidence	expected_elements vacío	A040	Missing expected elements
MethodSetSynthesis	método sin description	A050	Method metadata incomplete
RubricTransposition	min_score orden incorrecto	A060	Rubric thresholds out of order
MesoMacroEmbedding	cluster hermeticidad rota	A070	Cluster hermeticity violation
IntegritySealing	hash cálculo inconsistente	A080	Monolith hash mismatch
ValidationReport	conteo legacy ≠ destino	A090	Atom loss detected
FinalEmission	archivo size == 0	A100	Empty monolith emission
6. Estructura Definitiva del Monolito (Esqueleto)
JSON
{
  "version": "1.0.0",
  "generated_at": "2025-10-29T00:00:00Z",
  "integrity": {
    "monolith_hash": "<sha256>",
    "method_catalog_hash": "<sha256_metodos>",
    "enforcement": {
      "orchestrator_must_load": true,
      "disallow_direct_file_access": true
    }
  },
  "blocks": {
    "niveles_abstraccion": { ... },
    "orquestacion_secuencia": { ... },
    "indicadores_empiricos": { ... },
    "rubricacion_scoring": { ... }
  }
}
7. Arquitectura Interna del Agente
Módulos (clases) y responsabilidades:

Clase	Responsabilidad
LegacyLoader	Carga y sanea sources permitidos; produce raw dicts.
CanonicalIndexBuilder	Construye índices y verifica numeración y base_slots.
SlotMapper	Aplica fórmula base_slot y verifica cobertura ×10.
FieldNormalizer	Normaliza textos, modos, listas (sin alterar semántica).
EvidenceDistributor	Separa expected_elements/patterns/validations/evidence_expectations.
MethodSetInjector	Inyecta method_sets desde catálogo; valida schema parcial.
RubricTransposer	Transfiere modalidades y niveles; compone scoring_matrix.
MesoMacroAssembler	Inserta clusters y pregunta macro con reglas.
IntegritySealer	Serializa, calcula hash, fija integridad.
ValidationAuditor	Verifica invariantes y produce report detallado (forge_report.json).
MonolithEmitter	Emite archivos finales (monolito + manifest) de forma atómica.
AbortManager	Centraliza excepciones tipadas y códigos abort.
8. Formato de Reporte (forge_report.json)
JSON
{
  "legacy_counts": {
    "questions_micro": 300,
    "clusters_meso": 4,
    "question_macro": 1,
    "patterns_total": 451,
    "validation_checks_total": 300
  },
  "destination_counts": { ... identical ... },
  "loss_detected": false,
  "base_slot_coverage": {
    "D1-Q1": 10,
    "D1-Q2": 10
  },
  "method_set_summary": {
    "D1-Q1": { "methods": 18, "critical": 5 },
    "D6-Q5": { "methods": 22, "critical": 7 }
  },
  "rubric_thresholds_ok": true,
  "clusters_hermeticity_ok": true,
  "macro_rules": ["MACRO_CONVERGENTE_ALTO","MACRO_DISPERSION_CRITICA","MACRO_TRANSICION_MEDIO","MACRO_AMBIGUO"],
  "hash": "<sha256>",
  "timestamp": "2025-10-29T00:00:01Z"
}
9. Logging (forge.log)
Formato JSONL por fase:

JSON
{"phase":"LoadLegacy","status":"start","timestamp":"..."}
{"phase":"LoadLegacy","status":"ok","files":["questionnaire.json","rubric_scoring.json","metodos_completos_nivel3.json"],"elapsed_ms":42.3}
{"phase":"BaseSlotMapping","status":"ok","slots_verified":30,"elapsed_ms":3.7}
{"phase":"IntegritySealing","status":"ok","monolith_hash":"abc123...","elapsed_ms":12.9}
10. Código Esqueleto Principal

monolith_builder.py
import json, hashlib, time, re, copy, os
from pathlib import Path
from dataclasses import dataclass

ABORT_CODES = {
    "A001":"Missing or invalid legacy file",
11. Validación Externa (Script Independiente)

validate_monolith.py
import json, sys

REQUIRED_BLOCKS = ["niveles_abstraccion","orquestacion_secuencia","indicadores_empiricos","rubricacion_scoring"]

def abort(msg):
    print("ABORT:", msg); sys.exit(1)
12. Mecanismos Anti-Silencio
Conteo de cada lista original vs destino (patterns, expected_elements, validation_checks) → si difiere, ABORT.
Registro de diferencias (forge_diff.json) incluyendo si se añadió algún campo artificial (debe marcarse “sintetizado”: true).
13. Extensión Controlada (Cambios Futuro)
Para incorporar nueva modalidad (TYPE_G):

Añadir en legacy_rubric origen (o PR de actualización).
Reconstruir monolito con versión incrementada (1.0.1).
Validar que scoring_matrix contemple TYPE_G en policy areas que la permiten.
Generar diff de rubricacion_scoring y firmar cambio (hash nuevo). Sin seguir protocolo → ABORT en IntegritySealer.
14. Output Final (Artefactos Esperados)
questionnaire_monolith.json
forge_report.json
forge.log
forge_diff.json (si hubo síntesis)
validate_monolith.py (herramienta de verificación) Todos deben existir y ser coherentes.
15. Resumen Ultra-Compacto
MonolithForgeAgent = pipeline determinista de 11 fases con abort codes A001–A100; migración bit a bit sin pérdida; clusters herméticos; hash sellado; logs estructurados; invariantes irreductibles.
