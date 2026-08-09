# Contributing to AutoCircuit

AutoCircuit is an experimental mechanistic-interpretability research codebase. Contributions are welcome, but changes that alter scientific semantics need stronger review than ordinary software changes.

## Development setup

Use Python 3.11 or 3.12. For GPU work, install the correct PyTorch build for your system before installing the project dependencies.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

Before opening a pull request, run:

```bash
pytest -q
ruff check src tests
python -m mypy src/autocircuit
git diff --check
```

Model-loading/CUDA integration runs should be reported separately from offline unit-test results.

## Research-integrity rules

Please preserve the following invariants unless a pull request explicitly proposes a protocol revision:

1. **Do not inspect held-out data opportunistically.** Discovery, validation, and test populations have different roles.
2. **Do not weaken thresholds after seeing results.** Scientific gates and candidate-selection rules should be frozen before confirmatory evaluation.
3. **Do not silently filter difficult examples.** Processing failures, rejected examples, exclusions, and degenerate cases must remain visible in artifacts/reports.
4. **Do not call sensitivity a circuit.** Zero ablation, patching recovery, localization, necessity, sufficiency, and complete-circuit claims are distinct forms of evidence.
5. **Preserve provenance.** Derived runs should validate upstream configuration, model identity, artifact hashes, and relevant protocol versions.
6. **Prefer deterministic outputs.** Record random seeds and keep serialization stable when an artifact is intended to be reproducible.
7. **Fail closed.** If required provenance or invariants cannot be verified, stop and report the failure rather than continuing with ambiguous state.

The authoritative MVP protocol is [`docs/MVP_SPEC.md`](docs/MVP_SPEC.md).

## Pull requests

A useful research PR should state:

- the scientific or software question it addresses;
- whether it uses discovery, validation, or test data;
- exactly what interventions/metrics changed;
- what new artifacts are produced;
- offline checks run;
- any real model/CUDA run performed, including whether it was exploratory or confirmatory;
- what the result **does not** establish.

If the change modifies a frozen scientific decision rule, call that out prominently instead of presenting it as an implementation detail.

## Legacy code

The root repository also retains historical AI Safety Camp / Neuronpedia-oriented prototypes and manuscript material. New maintained MVP work should normally live under `src/autocircuit/`, `tests/`, `configs/`, and the current research docs. See [`docs/LEGACY.md`](docs/LEGACY.md).
