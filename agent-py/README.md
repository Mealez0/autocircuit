# Neuronpedia Attribution Graph Agent

This directory is a legacy-compatible CLI for generating Neuronpedia attribution graphs, cleaning them locally, and optionally labeling grouped nodes.

The API bridge is an actual transport layer: `generate-only` and `cleanup` execute HTTP requests against a Neuronpedia graph server. They do not print placeholder messages such as "would call the API" and then stop.

## Install

Minimal graph generation / cleanup:

```bash
python -m pip install -r requirements.txt
```

Optional LLM labels:

```bash
python -m pip install -r requirements-labeling.txt
```

Development tests:

```bash
python -m pip install -r requirements-dev.txt
PYTHONPATH=. pytest -q tests
```

The old semantic-grouping dependency stack was removed from the core install because semantic, layer, and hybrid grouping are not implemented. Those modes are now rejected explicitly instead of silently producing functional-grouping output.

## Graph server connection

The CLI targets the Neuronpedia graph-server contract (`POST /generate-graph` and `POST /steer`). The default is a local server on port 5004.

```bash
export NEURONPEDIA_GRAPH_URL=http://localhost:5004
export NEURONPEDIA_GRAPH_SECRET=your_server_secret
```

The default auth header is `x-secret-key`. For a proxy or deployment that uses another header:

```bash
export NEURONPEDIA_AUTH_HEADER=x-api-key
```

Secrets are sent only as request headers. They are not written into graph provenance manifests.

## Generate a graph

```bash
python main.py generate-only \
  --prompt "The capital of France is" \
  --model google/gemma-2-2b \
  --output graph.json
```

On success the command prints a small structured JSON result containing the graph path, response byte count, SHA-256, and provenance-manifest path.

Two files are written atomically:

- `graph.json` — the graph-server response
- `graph.json.manifest.json` — endpoint, non-secret request payload, response size, content type, and SHA-256

Large graph responses are streamed to disk instead of being buffered in memory.

## Generate and clean in one command

```bash
python main.py cleanup \
  --prompt "The capital of France is" \
  --raw-graph-output raw_graph.json \
  --output cleaned_graph.json \
  --strategy pathway \
  --grouping functional
```

The command performs the API request first and only enters local graph cleanup after a valid JSON-shaped response is written.

## Analyze or clean an existing graph

```bash
python main.py analyze --graph-file graph.json

python main.py cleanup-existing \
  --graph-file graph.json \
  --strategy balanced \
  --grouping functional \
  --output cleaned_graph.json
```

## Optional LLM labels

Labeling is opt-in. Install the labeling requirements and set:

```bash
export ANTHROPIC_API_KEY=your_key
```

Then pass the key through the environment when running `cleanup` or `cleanup-existing`.

Labeling failures are not converted into plausible generic text. The command exits with an explicit error so an agent/controller can distinguish success from failure and decide whether to retry or continue without labels.

## Retry behavior

Graph generation can be expensive, so automatic POST retries are disabled by default. Use `--retries N` only when the caller explicitly wants retry behavior. Retryable statuses are limited to 429, 502, 503, and 504 plus transport failures.

## Current cleanup strategies

Node selection:

- `pathway`
- `importance`
- `balanced`

Grouping:

- `functional` — implemented
- `semantic` — not implemented; explicit error
- `layer` — not implemented; explicit error
- `hybrid` — not implemented; explicit error

This fail-visible behavior prevents research output from being mislabeled as the result of an algorithm that never ran.

## Safety / failure semantics

The agent bridge follows fail-visible behavior:

- HTTP failures raise an explicit `NeuronpediaAPIError`.
- 2xx non-JSON graph responses are rejected.
- Empty responses are rejected.
- Raw graph files are atomically replaced only after a complete response is received.
- LLM labeling failures are surfaced instead of silently returning fabricated fallback labels.
- Authentication secrets are never included in provenance artifacts.
- Agent instructions explicitly prohibit simulated tool/API calls in prose.

This keeps safety checks without turning execution into a text-only simulation.
