"""Deterministic node selection for graph cleanup."""

from __future__ import annotations

from .graph_analyzer import GraphAnalyzer


class NodeSelector:
    """Select graph nodes to pin using deterministic heuristics."""

    SUPPORTED_STRATEGIES = frozenset({"pathway", "importance", "balanced"})

    def __init__(self, analyzer: GraphAnalyzer, max_nodes: int = 30) -> None:
        if max_nodes <= 0:
            raise ValueError("max_nodes must be positive")
        self.analyzer = analyzer
        self.max_nodes = max_nodes

    def select_nodes_for_pinning(self, strategy: str = "pathway") -> list[str]:
        if strategy == "pathway":
            return self._pathway_strategy()
        if strategy == "importance":
            return self._importance_strategy()
        if strategy == "balanced":
            return self._balanced_strategy()
        raise ValueError(f"Unknown strategy: {strategy}")

    @staticmethod
    def _append_unique(destination: list[str], seen: set[str], node_ids: list[str], limit: int) -> bool:
        """Append unseen IDs in encounter order; return True once the limit is reached."""
        for node_id in node_ids:
            if node_id in seen:
                continue
            seen.add(node_id)
            destination.append(node_id)
            if len(destination) >= limit:
                return True
        return False

    def _pathway_strategy(self) -> list[str]:
        """Select nodes from strongest paths while preserving deterministic path order."""
        input_features = self.analyzer.identify_input_features(layer_threshold=5)
        output_features = self.analyzer.identify_output_features(layer_threshold=16)
        paths = self.analyzer.trace_pathways(input_features[:10], output_features[:10])

        selected: list[str] = []
        seen: set[str] = set()
        for path in paths:
            if self._append_unique(selected, seen, path.bottleneck_nodes, self.max_nodes):
                break
            if self._append_unique(selected, seen, path.nodes, self.max_nodes):
                break
        return selected

    def _importance_strategy(self) -> list[str]:
        importance = self.analyzer.compute_node_importance()
        sorted_nodes = sorted(
            importance.items(),
            key=lambda item: (-item[1], item[0]),
        )
        return [node_id for node_id, _ in sorted_nodes[: self.max_nodes]]

    def _balanced_strategy(self) -> list[str]:
        """Select an approximately 30/40/30 early/middle/late layer mixture."""
        importance = self.analyzer.compute_node_importance()
        buckets: dict[str, list[tuple[str, float]]] = {
            "input": [],
            "middle": [],
            "output": [],
        }

        for node_id, score in importance.items():
            node = self.analyzer.get_node(node_id)
            if not node:
                continue
            layer = int(node.get("layer", 0))
            if layer <= 5:
                buckets["input"].append((node_id, score))
            elif layer <= 15:
                buckets["middle"].append((node_id, score))
            else:
                buckets["output"].append((node_id, score))

        for items in buckets.values():
            items.sort(key=lambda item: (-item[1], item[0]))

        quotas = {
            "input": int(self.max_nodes * 0.30),
            "middle": int(self.max_nodes * 0.40),
            "output": int(self.max_nodes * 0.30),
        }
        selected: list[str] = []
        seen: set[str] = set()
        for bucket in ("input", "middle", "output"):
            for node_id, _ in buckets[bucket][: quotas[bucket]]:
                if node_id not in seen:
                    selected.append(node_id)
                    seen.add(node_id)

        # Integer quotas and sparse layer buckets can under-fill the requested
        # budget. Fill the remainder from the strongest unselected nodes using a
        # deterministic score/id ordering.
        if len(selected) < self.max_nodes:
            ranked = sorted(importance.items(), key=lambda item: (-item[1], item[0]))
            for node_id, _ in ranked:
                if node_id in seen:
                    continue
                selected.append(node_id)
                seen.add(node_id)
                if len(selected) >= self.max_nodes:
                    break

        return selected
