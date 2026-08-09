# Neuronpedia Graph API — Execution Contract

Use this skill only when the task requires a Neuronpedia attribution graph or steering intervention.

## Non-negotiable execution rule

**Never simulate an API/tool call in prose.**

Forbidden examples:

- `[Makes API call ...]`
- `I would call /generate-graph ...`
- invented graph nodes, labels, completions, logits, status codes, URLs, or success messages

If execution is requested, one of these must happen:

1. Execute the real transport/CLI and use its returned artifact/result; or
2. Return an explicit execution error explaining what dependency, credential, server, or tool is unavailable.

A natural-language description of an intended call is never a substitute for execution.

## Preferred local bridge

Use the repository bridge instead of reconstructing HTTP requests in the agent prompt:

```bash
python agent-py/main.py generate-only \
  --prompt "The capital of France is" \
  --model google/gemma-2-2b \
  --output graph.json
```

Generate + clean:

```bash
python agent-py/main.py cleanup \
  --prompt "The capital of France is" \
  --model google/gemma-2-2b \
  --raw-graph-output raw_graph.json \
  --output cleaned_graph.json
```

The bridge uses `NEURONPEDIA_GRAPH_URL`, `NEURONPEDIA_GRAPH_SECRET`, and optionally `NEURONPEDIA_AUTH_HEADER`. Do not print or persist secrets.

## Current graph-server contract

The transport is built around these graph-server endpoints:

- `POST /generate-graph`
- `POST /steer`

The default local server is `http://localhost:5004`; the default authentication header is `x-secret-key`. Deployment-specific URL/header overrides belong in environment variables or CLI flags, not hard-coded agent prose.

## Result handling

### Graph generation

Treat generation as successful only when the bridge exits successfully and produces the requested graph artifact.

The bridge also creates `<graph>.manifest.json` containing non-secret request provenance, response size, content type, and SHA-256. Use that manifest when reporting reproducibility information.

Graph responses may be large. Do not load a large graph into the agent context unless the requested analysis actually needs its full contents. Prefer targeted programmatic analysis and concise summaries.

### Steering

Use the typed `NeuronpediaGraphClient.steer(...)` transport from `agent-py/neuronpedia_agent/api/client.py` when code execution is available. Consume the returned JSON object. Never invent missing fields or infer a successful intervention from a request payload alone.

## Failure semantics

Execution must be fail-visible:

- Authentication/HTTP failures are errors, not textual fallback results.
- Empty responses are errors.
- A successful HTTP status with a non-JSON graph body is an error.
- LLM-labeling failure is an error unless the caller explicitly chose to continue without labels.
- Do not silently switch API endpoints, models, grouping algorithms, datasets, prompts, or intervention parameters.
- Do not retry an expensive graph-generation POST unless retry behavior was explicitly enabled.

When an operation fails, report the concrete error and preserve any safe diagnostic information. Do not claim that a graph, intervention, label, or artifact exists unless it was actually produced.

## Research-integrity boundary

This API skill does not override repository research controls in `AGENTS.md`.

In particular:

- Do not inspect or use protected held-out/test data for exploratory work.
- Do not reinterpret a failed preregistered gate as confirmation.
- Do not convert an exploratory graph/steering observation into a circuit claim without the required causal and held-out evidence.
- Preserve model, prompt, intervention, seed, and artifact provenance when those fields are relevant to the experiment.

## Interpretation rule

Interpret only observed output. Separate:

- **observed** API/model results,
- **computed** graph statistics,
- **inference/hypothesis** about mechanism.

Do not describe attribution graphs as literal hidden chain-of-thought. They are mechanistic-analysis artifacts and causal hypotheses require intervention evidence.

## Context discipline

Keep agent context small:

- use the CLI/transport for execution,
- use artifact paths and manifests for provenance,
- inspect only the graph sections needed for the current question,
- avoid pasting large raw JSON into chat,
- avoid repeating endpoint documentation when the transport already encodes it.

The agent's job is to execute, verify, and interpret — not to role-play execution.
