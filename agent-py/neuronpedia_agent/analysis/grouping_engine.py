"""Grouping primitives for turning pinned graph nodes into supernodes."""

from __future__ import annotations

from dataclasses import dataclass

from .graph_analyzer import GraphAnalyzer


@dataclass
class Supernode:
    """A group of nodes with a similar coarse functional role."""

    label: str
    node_ids: list[str]
    layer_range: tuple[int, int]
    functional_role: str
    total_influence: float


class GroupingEngine:
    """Group pinned nodes into interpretable supernodes.

    Only ``functional`` grouping is implemented. Other historical strategy names
    are rejected explicitly instead of silently falling back to a different
    algorithm and producing misleading research output.
    """

    SUPPORTED_STRATEGIES = frozenset({"functional"})

    def __init__(self, analyzer: GraphAnalyzer, pinned_nodes: list[str]) -> None:
        self.analyzer = analyzer
        self.pinned_nodes = pinned_nodes

    def create_supernodes(self, strategy: str = "functional") -> list[Supernode]:
        if strategy not in self.SUPPORTED_STRATEGIES:
            raise NotImplementedError(
                f"Grouping strategy {strategy!r} is not implemented; "
                "use 'functional' until a tested implementation is added"
            )
        return self._functional_grouping()

    def _layer(self, node_id: str) -> int:
        node = self.analyzer.get_node(node_id)
        if not node:
            return 0
        return int(node.get("layer", 0))

    def _outgoing_influence(self, node_ids: list[str]) -> float:
        node_set = set(node_ids)
        return sum(
            float(edge["weight"])
            for edge in self.analyzer.edges
            if edge["source"] in node_set
        )

    def _make_supernode(self, role: str, node_ids: list[str]) -> Supernode:
        layers = [self._layer(node_id) for node_id in node_ids]
        return Supernode(
            label="",
            node_ids=list(node_ids),
            layer_range=(min(layers), max(layers)),
            functional_role=role,
            total_influence=self._outgoing_influence(node_ids),
        )

    def _functional_grouping(self) -> list[Supernode]:
        roles: dict[str, list[str]] = {
            "input_detector": [],
            "relational_processor": [],
            "output_promoter": [],
        }

        for node_id in self.pinned_nodes:
            node = self.analyzer.get_node(node_id)
            if not node:
                continue

            layer = int(node.get("layer", 0))
            if layer >= 16:
                logit_influence = sum(
                    float(edge["weight"])
                    for edge in self.analyzer.edges
                    if edge["source"] == node_id
                    and str(edge["target"]).startswith("logit_")
                )
                if logit_influence > 0.1:
                    roles["output_promoter"].append(node_id)
                    continue

            if layer <= 5:
                roles["input_detector"].append(node_id)
            else:
                roles["relational_processor"].append(node_id)

        supernodes: list[Supernode] = []
        for role, node_ids in roles.items():
            if not node_ids:
                continue
            if len(node_ids) <= 3:
                supernodes.append(self._make_supernode(role, node_ids))
                continue

            sorted_nodes = sorted(node_ids, key=self._layer)
            current_group = [sorted_nodes[0]]
            current_layer = self._layer(sorted_nodes[0])

            for node_id in sorted_nodes[1:]:
                node_layer = self._layer(node_id)
                if node_layer - current_layer <= 3:
                    current_group.append(node_id)
                    continue

                supernodes.append(self._make_supernode(role, current_group))
                current_group = [node_id]
                current_layer = node_layer

            if current_group:
                supernodes.append(self._make_supernode(role, current_group))

        return supernodes
