# Maintained causal-research MVP instructions

## Scientific data boundaries

- Never open, read, generate, inspect, score, summarize, or otherwise access the untouched test split.
- Never reuse held-out validation examples for exploratory discovery, localization, candidate selection, hyperparameter selection, or debugging.
- Previously produced aggregate validation reports may be read only for historical reporting and provenance checks.
- Use only frozen discovery artifacts for exploratory mechanism localization.
- Never lower, alter, reinterpret, bypass, or rerun preregistered gates after observing their outcomes.
- The frozen held-out validation result remains `HELD_OUT_VALIDATION_NOT_CONFIRMED`.
- Never reinterpret the failed frozen validation gate as confirmation.

## Statistical rules

- Preserve the matched family as the statistical unit.
- Do not replace family-level statistics with example-level independence assumptions.
- Use deterministic seeds.
- Use deterministic matched-family bootstrap intervals.
- Preserve clean/corrupt, bidirectional, matched-source, and family-permuted-source controls unless a reviewed protocol explicitly changes them.

## Provenance and artifact safety

- Verify upstream manifests, hashes, model identity, protocol versions, selected-layer provenance, and completed-run status.
- Never silently overwrite completed artifacts.
- Never delete or replace artifacts without an explicit `--force` operation.
- Mark incomplete runs clearly and retain enough diagnostic information to investigate failure.

## Scientific interpretation

- Localization alone is not a circuit claim.
- A circuit claim requires evidence addressing:
  - necessity;
  - sufficiency;
  - specificity;
  - appropriate null controls;
  - robustness;
  - genuinely held-out evidence.
- Keep:
  - `scientific_confirmation: false`
  - `circuit_found: false`
  until those requirements are actually met.

## Engineering workflow

- Do not run CUDA experiments until offline tests and static checks pass.
- Every implementation task must run:
  - `pytest -q`
  - `ruff check src tests`
  - `python -m mypy src/autocircuit`
  - `git diff --check`
- Keep changes small and reviewable.
- Avoid unrelated refactors.
- Do not install, upgrade, or remove dependencies unless explicitly requested.
- Do not commit, push, merge, or open a pull request unless explicitly requested.
