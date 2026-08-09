from __future__ import annotations

from dataclasses import dataclass

from neuronpedia_agent.analysis.node_selector import NodeSelector


@dataclass
class FakePath:
    nodes: list[str]
    bottleneck_nodes: list[str]


class FakeAnalyzer:
    def identify_input_features(self, layer_threshold: int = 5) -> list[str]:
        del layer_threshold
        return ["input-a", "input-b"]

    def identify_output_features(self, layer_threshold: int = 16) -> list[str]:
        del layer_threshold
        return ["output-a"]

    def trace_pathways(self, source_nodes: list[str], target_nodes: list[str]) -> list[FakePath]:
        assert source_nodes == ["input-a", "input-b"]
        assert target_nodes == ["output-a"]
        return [
            FakePath(nodes=["input-a", "middle-a", "output-a"], bottleneck_nodes=["middle-a"]),
            FakePath(nodes=["input-b", "middle-a", "output-a"], bottleneck_nodes=["input-b"]),
        ]

    def compute_node_importance(self) -> dict[str, float]:
        return {"b": 0.5, "a": 0.5, "c": 0.2}

    def get_node(self, node_id: str) -> dict[str, int]:
        return {
            "a": {"layer": 2},
            "b": {"layer": 10},
            "c": {"layer": 20},
        }[node_id]


def test_pathway_selection_preserves_first_seen_order() -> None:
    selector = NodeSelector(FakeAnalyzer(), max_nodes=4)  # type: ignore[arg-type]

    first = selector.select_nodes_for_pinning("pathway")
    second = selector.select_nodes_for_pinning("pathway")

    assert first == ["middle-a", "input-a", "output-a", "input-b"]
    assert second == first


def test_importance_ties_are_broken_by_node_id() -> None:
    selector = NodeSelector(FakeAnalyzer(), max_nodes=3)  # type: ignore[arg-type]

    assert selector.select_nodes_for_pinning("importance") == ["a", "b", "c"]


def test_balanced_strategy_fills_rounding_remainder_deterministically() -> None:
    selector = NodeSelector(FakeAnalyzer(), max_nodes=3)  # type: ignore[arg-type]

    selected = selector.select_nodes_for_pinning("balanced")

    assert selected == ["b", "a", "c"]
    assert len(selected) == 3
