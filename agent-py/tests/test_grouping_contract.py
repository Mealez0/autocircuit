from __future__ import annotations

import pytest

from neuronpedia_agent.analysis.graph_analyzer import GraphAnalyzer
from neuronpedia_agent.analysis.grouping_engine import GroupingEngine


def _analyzer() -> GraphAnalyzer:
    return GraphAnalyzer(
        {
            "nodes": [
                {"id": "early", "layer": 2},
                {"id": "middle", "layer": 10},
                {"id": "late", "layer": 20},
            ],
            "edges": [
                {"source": "early", "target": "middle", "weight": 0.4},
                {"source": "middle", "target": "late", "weight": 0.5},
                {"source": "late", "target": "logit_target", "weight": 0.6},
            ],
        }
    )


def test_functional_grouping_is_the_only_implemented_strategy() -> None:
    engine = GroupingEngine(_analyzer(), ["early", "middle", "late"])

    groups = engine.create_supernodes("functional")

    assert {group.functional_role for group in groups} == {
        "input_detector",
        "relational_processor",
        "output_promoter",
    }


@pytest.mark.parametrize("strategy", ["semantic", "layer", "hybrid"])
def test_unimplemented_grouping_never_silently_falls_back(strategy: str) -> None:
    engine = GroupingEngine(_analyzer(), ["early", "middle", "late"])

    with pytest.raises(NotImplementedError, match="not implemented"):
        engine.create_supernodes(strategy)
