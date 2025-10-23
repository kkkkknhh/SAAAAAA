from datetime import datetime, timezone

import pytest

pytest.importorskip("networkx")
pytest.importorskip("numpy")
pytest.importorskip("scipy")

from teoria_cambio import AdvancedGraphNode, AdvancedDAGValidator


def test_advanced_graph_node_serialization_defaults_and_sorting():
    node = AdvancedGraphNode(
        name="autonomia_economica",
        dependencies={"reduccion_vbg", "aumento_participacion"},
        metadata={"confidence": "0.75", "note": {"fuente": "Plan 2024"}},
        role="RESULTADO",
    )

    payload = node.to_serializable_dict()

    assert payload["name"] == "autonomia_economica"
    assert payload["dependencies"] == [
        "aumento_participacion",
        "reduccion_vbg",
    ]
    assert isinstance(payload["metadata"]["created"], str)
    assert payload["metadata"]["confidence"] == pytest.approx(0.75)
    assert payload["metadata"]["note"] == "{'fuente': 'Plan 2024'}"
    assert payload["role"] == "resultado"


def test_export_nodes_enforces_schema_and_populates_defaults():
    validator = AdvancedDAGValidator()
    validator.add_node("recursos_financieros", role="insumo")
    validator.add_node(
        "reduccion_vbg",
        dependencies={"recursos_financieros"},
        metadata={
            "created": datetime(2024, 1, 1, tzinfo=timezone.utc),
            "confidence": 1.2,
            "evidencia": ["Encuesta 2023"],
        },
        role="resultado",
    )

    nodes = validator.export_nodes(validate=True)

    assert len(nodes) == 2
    resultado_node = next(node for node in nodes if node["name"] == "reduccion_vbg")
    assert resultado_node["dependencies"] == ["recursos_financieros"]
    assert resultado_node["metadata"]["confidence"] == pytest.approx(1.0)
    assert resultado_node["metadata"]["evidencia"] == "['Encuesta 2023']"
    assert resultado_node["metadata"]["created"].startswith("2024-01-01")
    # Defaults populated for nodes without explicit metadata
    insumo_node = next(node for node in nodes if node["name"] == "recursos_financieros")
    assert "created" in insumo_node["metadata"]
    assert insumo_node["metadata"]["confidence"] == pytest.approx(1.0)
    assert validator.last_serialized_nodes == nodes


def test_advanced_graph_node_rejects_unknown_role():
    with pytest.raises(ValueError):
        AdvancedGraphNode(name="actor_externo", dependencies=set(), role="actor")
