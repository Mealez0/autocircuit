"""CLI for Neuronpedia attribution-graph generation and cleanup."""

from __future__ import annotations

import json
from pathlib import Path

import click

from neuronpedia_agent.analysis.graph_analyzer import GraphAnalyzer
from neuronpedia_agent.analysis.grouping_engine import GroupingEngine
from neuronpedia_agent.analysis.node_selector import NodeSelector
from neuronpedia_agent.api import NeuronpediaAPIError, NeuronpediaGraphClient
from neuronpedia_agent.labeling.auto_labeler import AutoLabeler

DEFAULT_MODEL = "google/gemma-2-2b"
DEFAULT_API_URL = "http://localhost:5004"


@click.group()
def cli() -> None:
    """Neuronpedia Attribution Graph Cleanup Automation Agent."""


def _clean_graph_file(
    graph_file: str | Path,
    output: str | Path,
    *,
    strategy: str,
    max_nodes: int,
    grouping: str,
    anthropic_api_key: str | None,
) -> dict[str, object]:
    source = Path(graph_file)
    with source.open("r", encoding="utf-8") as handle:
        graph_data = json.load(handle)

    if not isinstance(graph_data, dict):
        raise click.ClickException("Graph JSON must be an object")

    analyzer = GraphAnalyzer(graph_data)
    nodes = graph_data.get("nodes", [])
    edges = graph_data.get("edges", [])
    click.echo(f"Graph loaded: {len(nodes)} nodes, {len(edges)} edges")

    selector = NodeSelector(analyzer, max_nodes=max_nodes)
    pinned_nodes = selector.select_nodes_for_pinning(strategy=strategy)
    click.echo(f"Selected {len(pinned_nodes)} nodes using {strategy} strategy")

    grouper = GroupingEngine(analyzer, pinned_nodes)
    supernodes = grouper.create_supernodes(strategy=grouping)
    click.echo(f"Created {len(supernodes)} supernodes using {grouping} grouping")

    if anthropic_api_key:
        labeler = AutoLabeler(api_key=anthropic_api_key)
        for supernode in supernodes:
            supernode.label = labeler.generate_label(supernode, graph_data, prompt="", target_logit="")

    output_data: dict[str, object] = {
        "pinned_node_ids": pinned_nodes,
        "supernodes": [
            {
                "label": supernode.label,
                "node_ids": supernode.node_ids,
                "layer_range": supernode.layer_range,
                "functional_role": supernode.functional_role,
                "total_influence": supernode.total_influence,
            }
            for supernode in supernodes
        ],
        "original_graph": graph_data,
    }

    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(f".{destination.name}.tmp")
    try:
        with tmp.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(output_data, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        tmp.replace(destination)
    finally:
        if tmp.exists():
            tmp.unlink()

    click.echo(f"✓ Saved cleaned graph to: {destination}")
    return output_data


def _make_client(
    api_url: str,
    api_secret: str | None,
    auth_header: str,
    timeout: float,
    retries: int,
) -> NeuronpediaGraphClient:
    return NeuronpediaGraphClient(
        base_url=api_url,
        secret=api_secret,
        auth_header=auth_header,
        timeout_seconds=timeout,
        retries=retries,
    )


@cli.command()
@click.option("--prompt", required=True, help="Input prompt for graph generation")
@click.option("--model", default=DEFAULT_MODEL, show_default=True, help="Graph-server model id")
@click.option("--output", default="cleaned_graph.json", show_default=True, help="Cleaned graph output")
@click.option("--raw-graph-output", default="raw_graph.json", show_default=True, help="Raw API graph JSON")
@click.option(
    "--strategy",
    default="pathway",
    show_default=True,
    type=click.Choice(["pathway", "importance", "balanced"]),
)
@click.option("--max-nodes", default=30, show_default=True, type=int, help="Maximum nodes to pin")
@click.option(
    "--grouping",
    default="functional",
    show_default=True,
    type=click.Choice(["functional", "semantic", "layer", "hybrid"]),
)
@click.option("--api-url", envvar="NEURONPEDIA_GRAPH_URL", default=DEFAULT_API_URL, show_default=True)
@click.option("--api-secret", envvar="NEURONPEDIA_GRAPH_SECRET", help="Graph server secret")
@click.option("--auth-header", envvar="NEURONPEDIA_AUTH_HEADER", default="x-secret-key", show_default=True)
@click.option("--timeout", default=600.0, show_default=True, type=float)
@click.option("--retries", default=0, show_default=True, type=int)
@click.option("--anthropic-api-key", envvar="ANTHROPIC_API_KEY", help="Optional labeling API key")
def cleanup(
    prompt: str,
    model: str,
    output: str,
    raw_graph_output: str,
    strategy: str,
    max_nodes: int,
    grouping: str,
    api_url: str,
    api_secret: str | None,
    auth_header: str,
    timeout: float,
    retries: int,
    anthropic_api_key: str | None,
) -> None:
    """Generate a graph through the API, then clean it locally."""
    client = _make_client(api_url, api_secret, auth_header, timeout, retries)
    click.echo(f"Generating graph for: {prompt}")
    click.echo(f"Graph server: {client.base_url}")

    try:
        generated = client.generate_graph(
            prompt=prompt,
            model_id=model,
            output_path=raw_graph_output,
        )
        click.echo(f"✓ API graph saved: {generated.path} ({generated.byte_count:,} bytes)")
        _clean_graph_file(
            generated.path,
            output,
            strategy=strategy,
            max_nodes=max_nodes,
            grouping=grouping,
            anthropic_api_key=anthropic_api_key,
        )
    except NeuronpediaAPIError as exc:
        detail = f" ({exc.body})" if exc.body else ""
        raise click.ClickException(f"Neuronpedia API error: {exc}{detail}") from exc


@cli.command("cleanup-existing")
@click.option("--graph-file", required=True, type=click.Path(exists=True), help="Path to graph JSON")
@click.option("--output", default="cleaned_graph.json", show_default=True, help="Output file path")
@click.option(
    "--strategy",
    default="pathway",
    show_default=True,
    type=click.Choice(["pathway", "importance", "balanced"]),
)
@click.option("--max-nodes", default=30, show_default=True, type=int)
@click.option(
    "--grouping",
    default="functional",
    show_default=True,
    type=click.Choice(["functional", "semantic", "layer", "hybrid"]),
)
@click.option("--api-key", envvar="ANTHROPIC_API_KEY", help="Anthropic API key for optional labeling")
def cleanup_existing(
    graph_file: str,
    output: str,
    strategy: str,
    max_nodes: int,
    grouping: str,
    api_key: str | None,
) -> None:
    """Clean an existing graph JSON without making a graph-server call."""
    try:
        _clean_graph_file(
            graph_file,
            output,
            strategy=strategy,
            max_nodes=max_nodes,
            grouping=grouping,
            anthropic_api_key=api_key,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise click.ClickException(str(exc)) from exc


@cli.command("generate-only")
@click.option("--prompt", required=True)
@click.option("--model", default=DEFAULT_MODEL, show_default=True)
@click.option("--output", default="graph.json", show_default=True)
@click.option("--api-url", envvar="NEURONPEDIA_GRAPH_URL", default=DEFAULT_API_URL, show_default=True)
@click.option("--api-secret", envvar="NEURONPEDIA_GRAPH_SECRET", help="Graph server secret")
@click.option("--auth-header", envvar="NEURONPEDIA_AUTH_HEADER", default="x-secret-key", show_default=True)
@click.option("--timeout", default=600.0, show_default=True, type=float)
@click.option("--retries", default=0, show_default=True, type=int)
def generate_only(
    prompt: str,
    model: str,
    output: str,
    api_url: str,
    api_secret: str | None,
    auth_header: str,
    timeout: float,
    retries: int,
) -> None:
    """Generate a graph through the API and persist the returned JSON."""
    client = _make_client(api_url, api_secret, auth_header, timeout, retries)
    try:
        generated = client.generate_graph(prompt=prompt, model_id=model, output_path=output)
    except NeuronpediaAPIError as exc:
        detail = f" ({exc.body})" if exc.body else ""
        raise click.ClickException(f"Neuronpedia API error: {exc}{detail}") from exc

    click.echo(
        json.dumps(
            {
                "status": "ok",
                "path": str(generated.path),
                "bytes": generated.byte_count,
                "content_type": generated.content_type,
            },
            sort_keys=True,
        )
    )


@cli.command()
@click.option("--graph-file", required=True, type=click.Path(exists=True))
def analyze(graph_file: str) -> None:
    """Analyze a graph and print structural statistics without modifying it."""
    try:
        with open(graph_file, "r", encoding="utf-8") as handle:
            graph_data = json.load(handle)
        if not isinstance(graph_data, dict):
            raise click.ClickException("Graph JSON must be an object")

        analyzer = GraphAnalyzer(graph_data)
        num_nodes = len(graph_data.get("nodes", []))
        num_edges = len(graph_data.get("edges", []))
        click.echo("\nGraph Statistics:")
        click.echo(f"  Nodes: {num_nodes}")
        click.echo(f"  Edges: {num_edges}")

        layers = [node.get("layer", 0) for node in graph_data.get("nodes", [])]
        if layers:
            click.echo(f"  Layers: {min(layers)}-{max(layers)}")

        importance = analyzer.compute_node_importance()
        top_nodes = sorted(importance.items(), key=lambda item: item[1], reverse=True)[:10]
        click.echo("\nTop 10 Most Important Nodes:")
        for index, (node_id, score) in enumerate(top_nodes, 1):
            node = analyzer.get_node(node_id)
            explanation = node.get("explanation", "No explanation") if node else "Unknown"
            click.echo(f"  {index}. {node_id} (influence: {score:.3f}) - {explanation[:50]}")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise click.ClickException(str(exc)) from exc


if __name__ == "__main__":
    cli()
