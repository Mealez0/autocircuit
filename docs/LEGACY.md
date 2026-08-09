# Legacy and historical material

This repository contains two distinct generations of work under the AutoCircuit name.

## Maintained MVP

The maintained research code is the Pythia-70M causal circuit-discovery MVP in:

- `src/autocircuit/`
- `configs/`
- `tests/`
- `docs/MVP_SPEC.md`

This is the code described by the current root `README.md`.

## Historical AI Safety Camp / Neuronpedia work

Earlier work in this repository explored automated analysis of attribution graphs, agent-assisted feature/circuit annotation, Neuronpedia integration, graph mining, and research-paper tooling. That material is retained for provenance and may still be useful as a source of ideas or prototypes, but it is **not the maintained MVP and should not be treated as the current execution path**.

Historical/prototype areas include, among others:

- `agent-py/`
- `graph-analysis/`
- `experiments/`
- `claude-code-skills/`
- older manuscripts, figures, and paper-build utilities under `docs/`
- the GitHub Pages manuscript build

Some documents refer to the original AI Safety Camp team, project lead, or broader long-term plans. Those statements describe that earlier project context rather than ownership/status of the current maintained MVP.

## Why keep it?

The older material is preserved because it records research history, design ideas, experiments, and manuscript work that may be useful when the maintained causal-discovery pipeline later expands toward graph mining or automated interpretation.

New code should not depend on legacy directories unless a deliberate migration is being performed and tested.
