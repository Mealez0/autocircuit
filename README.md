# AutoCircuit

**Reproducible causal circuit discovery experiments for transformer language models.**

[![CI](https://github.com/Mealez0/autocircuit/actions/workflows/ci.yml/badge.svg)](https://github.com/Mealez0/autocircuit/actions/workflows/ci.yml)
[![Python 3.11–3.12](https://img.shields.io/badge/python-3.11%E2%80%933.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

AutoCircuit is an experimental mechanistic-interpretability research codebase for **automating the path from a controlled behavior to causal localization**. The maintained MVP currently targets synthetic associative recall in **Pythia-70M** using deterministic datasets, clean/corrupt contrasts, activation patching, matched controls, provenance checks, and held-out research guardrails.

> **Research status:** the current causal-localization results are exploratory and discovery-population only. They do **not** establish a complete circuit, necessity/sufficiency, or scientific confirmation. The untouched test split has not been used for the exploratory localization results described below.

## Current result

The current pipeline has progressed beyond scaffolding into real CUDA experiments on Pythia-70M:

1. A matched query-position study isolated a discovery-population position effect while controlling lexical/token differences.
2. Residual-stream activation patching localized the strongest family-specific transfer to **block 5**.
3. Block decomposition showed that **both attention and MLP outputs** carry the transfer signal.
4. Attention-head localization concentrated the attention contribution primarily in **heads 3 and 6**.
5. Head-set analysis found the pair **(3, 6)** ranked first among all 28 unordered head pairs in the frozen discovery population.

### Exploratory causal-localization snapshot

| Stage | Main observation |
|---|---|
| Residual boundary scan | `blocks.5.hook_resid_post` ranked first; matched-minus-permuted transfer advantage ≈ **0.607** |
| Block decomposition | MLP advantage ≈ **0.171**; attention advantage ≈ **0.153** |
| Individual heads | `head_3` advantage ≈ **0.083**; `head_6` ≈ **0.076** |
| Head pair | `(head_3, head_6)` matched transfer ≈ **0.339**; family-specific advantage ≈ **0.159** |
| Pair ranking | **1 / 28** for pair patching and leave-pair-out diagnostics |

For the head-pair run, the pair recovered approximately the aggregate attention transfer under the project metric and produced the largest leave-pair-out transfer loss. This is evidence for a **strong exploratory candidate**, not a claim that a complete circuit has been discovered.

See the merged research steps in [PR #7](https://github.com/Mealez0/autocircuit/pull/7), [#8](https://github.com/Mealez0/autocircuit/pull/8), [#9](https://github.com/Mealez0/autocircuit/pull/9), and [#10](https://github.com/Mealez0/autocircuit/pull/10).

## What AutoCircuit currently implements

- deterministic associative-recall dataset generation;
- clean/corrupt paired examples and target-vs-distractor logit metrics;
- discovery/validation/test separation with explicit split firewalls;
- reproducible baseline evaluation with TransformerLens;
- discovery diagnostics and preregistered format-candidate search;
- matched query-position studies;
- residual-stream layer localization with matched and permuted-family controls;
- attention-vs-MLP component localization;
- individual attention-head localization;
- attention head-set patching, leave-out analysis, interaction diagnostics, and specificity controls;
- deterministic bootstrap summaries;
- artifact hashing, provenance verification, resumable runs, and fail-closed checks;
- unit tests designed to run without downloading a model.

The maintained implementation lives in [`src/autocircuit/`](src/autocircuit/). The frozen MVP research contract is in [`docs/MVP_SPEC.md`](docs/MVP_SPEC.md).

## Research contract

The project deliberately separates **software success** from **scientific success**.

The MVP uses a synthetic associative-recall / induction-like key-value retrieval task because it provides exact targets, unlimited controlled examples, and a clean causal contrast. Discovery, validation, and test populations are separated, thresholds are frozen before confirmatory evaluation, and an empty/rejected circuit is treated as a valid scientific outcome.

The intended final protocol includes:

- clean-to-corrupt activation patching;
- held-out validation and untouched final test evaluation;
- matched random-component null controls;
- multiple-testing correction;
- effect-size and directionality requirements;
- explicit reporting of rejected candidates and failed baselines.

Read the exact decision rules in [`docs/MVP_SPEC.md`](docs/MVP_SPEC.md).

## Quick start

### Requirements

- Python **3.11–3.12**
- PyTorch with the appropriate CPU/CUDA build
- `transformer-lens==2.15.4`
- `transformers==4.51.3`

TransformerLens/Transformers are intentionally pinned to a validated pair. Transformers 5.x is rejected by the runtime compatibility check.

### Windows + NVIDIA GPU

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# Install a CUDA-enabled PyTorch build appropriate for your system first.
python -m pip install -e ".[dev]" --no-deps
python -m pip install packaging transformer-lens==2.15.4 transformers==4.51.3 pytest ruff mypy

python -m autocircuit.doctor
pytest -q
python -m autocircuit.smoke --model pythia-70m --device auto
```

### Linux / CPU development

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'

python -m autocircuit.doctor
pytest -q
ruff check src tests
python -m mypy src/autocircuit
```

Unit tests do not intentionally download Pythia-70M. Integration/model runs are separate.

## Main workflows

### Discovery pipeline

```bash
python -m autocircuit.pipeline discovery --device cuda
```

This regenerates the discovery population, evaluates the baseline, records per-example outputs, runs deterministic diagnostics, and evaluates preregistered task-format candidates. Discovery does not inspect validation or test data.

### Matched position study

```bash
python -m autocircuit.pipeline position-study --device cuda
```

This studies first/interior/last query positions on matched families while controlling lexical assignments and prompt/token-length structure.

### Causal localization

The causal-localization modules are currently explicit research stages rather than a single “discover everything” command:

```bash
python -m autocircuit.position_localization --help
python -m autocircuit.position_component_localization --help
python -m autocircuit.position_head_localization --help
python -m autocircuit.position_head_set_analysis --help
```

Each stage validates upstream provenance before loading the model and records structured artifacts for reproducibility.

## Repository map

```text
src/autocircuit/      Maintained MVP implementation
configs/              Frozen/reviewable research configuration
tests/                Offline-heavy regression and research-integrity tests
docs/MVP_SPEC.md      Frozen MVP scientific contract
docs/                 Manuscripts, figures, and historical project material
experiments/          Earlier exploratory experiments / prototypes
graph-analysis/       Earlier graph-analysis work
agent-py/             Legacy/prototype agent work
claude-code-skills/   Legacy/prototype automation material
```

Older AI Safety Camp / Neuronpedia-oriented work remains in the repository for provenance, but it is **not the maintained AutoCircuit MVP**. See [`docs/LEGACY.md`](docs/LEGACY.md).

## Reproducibility and integrity

AutoCircuit intentionally contains stronger guardrails than a typical research prototype:

- deterministic seeds and byte-stable serialized datasets;
- SHA-256 hashes for reusable artifacts;
- resumable runs with fingerprint/hash verification;
- explicit discovery/validation/test lifecycle separation;
- matched-family and permuted-family causal controls;
- exact identity/no-op diagnostics in causal stages;
- failure manifests instead of silently continuing after invalid state;
- `scientific_confirmation: false` / `circuit_found: false` until the required evidence exists.

The latest merged head-set analysis reported **167 passing tests** together with Ruff and strict mypy checks on that exact research branch before merge.

## Roadmap

Near-term work is focused on converting the current exploratory candidate into a defensible circuit claim or rejecting it:

- [ ] run the frozen held-out confirmatory stages where preregistered;
- [ ] expand causal analysis beyond the current attention head pair;
- [ ] test necessity/sufficiency with appropriately scoped interventions;
- [ ] add matched random-component null distributions for candidate circuits;
- [ ] perform final untouched test evaluation only after the candidate/protocol is frozen;
- [ ] produce a compact reproducible research report from generated artifacts;
- [ ] only then generalize the pipeline to additional behaviors/models.

## Contributing

This is a research codebase, so changes that affect datasets, metrics, intervention semantics, split access, or scientific decision rules require more scrutiny than ordinary refactors. See [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Historical context

AutoCircuit contains material from an earlier AI Safety Camp project centered on large-scale attribution-graph analysis and Neuronpedia. The current maintained MVP is a narrower causal-discovery program built around Pythia-70M and TransformerLens. Historical files are retained for provenance rather than presented as current functionality.

## License

MIT. See [`LICENSE`](LICENSE).
