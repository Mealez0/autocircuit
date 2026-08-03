# AutoCircuit Associative-Recall MVP: Frozen Research Contract

## Objective and rationale

The first behavior is **synthetic associative recall / induction-like key–value retrieval** on
Pythia-70M. Each prompt parametrically samples distinct names (keys), objects (values), fact order,
query key, and distractor facts. For example, facts may establish that John likes apples and Mary
likes books, followed by `John likes`, where the next-token target is ` apples`. This constrained
behavior has an exact answer, supports unlimited controlled examples, and admits a causal clean /
corrupt contrast. It is consequently a better first systems test than open-ended semantic behavior.

## Examples and splits

- A **clean** example contains a sampled bijection from keys to values and queries one key. Its
  target is that key's value; a distractor is another value present in the same prompt.
- Its **corrupt** partner keeps template, keys, values, fact count, query position, and token-length
  constraints matched, but swaps the queried key's value with the selected distractor. Thus the
  corrupt target is the clean distractor. Pairs with unequal relevant tokenization are rejected.
- Generation is deterministic from a recorded seed. No exact prompt, key/value assignment, or
  random seed is shared across splits. **Discovery** (default 256 pairs) ranks components and sets
  thresholds; **validation** (128) selects candidates without changing thresholds; **test** (128)
  is opened once for the final unbiased estimate. Split sizes and seed live in `configs/mvp.toml`.

Before intervention, a split is eligible only if at least 80% of examples have positive clean
target-vs-distractor logit difference and their mean difference is at least 1.0. Failure stops
discovery and is reported rather than silently filtering hard prompts.

## Discovery-diagnostic stage

The version-2 search is confined to discovery data. The orchestrator records every example and
reports performance by template, fact count, presented query position, token length, answer pair,
entity, and correctness quadrant, with deterministic bootstrap intervals and explicit underpowered
labels. It then generates fresh datasets for a compact source-controlled candidate registry. These
are population rules (fact range, relation template, query position, and separator), never
output-selected examples or identities.

The eligibility thresholds above remain frozen. Among eligible candidates the deterministic order
is clean accuracy, mean clean logit difference, clean/corrupt contrast, and lexical candidate ID.
No eligible candidate is a valid scientific stopping result rather than a software error. Neither
validation nor test is opened in this stage, and a discovery-selected v2 still requires held-out
evaluation. Activation patching and causal circuit discovery are explicitly deferred until the
behavioral gate is satisfied.

## Metric and intervention protocol

The primary per-example metric at the final query position is
`LD = logit(clean target) - logit(clean distractor)`. Multi-token answers are outside this MVP.
The future primary intervention is **clean-to-corrupt activation patching**: run corrupt prompts,
replace one component activation with its position-matched clean activation, and measure recovery
`(LD_patched - LD_corrupt) / (LD_clean - LD_corrupt)`. Degenerate denominators are excluded by a
predeclared epsilon and counted in the report.

The component search space is every individual attention-head output (`z`, separated by layer and
head) and every layer MLP output, at preregistered token positions. Zero ablation is only a
**sensitivity control**; it is not evidence that a component carries the clean/corrupt causal signal.

## Null control and decision rule

Each proposed circuit is compared with at least 1,000 random component sets matched for component
count and attention-head/MLP composition. Random seeds are recorded. A candidate is accepted only
if all of the following hold:

1. baseline eligibility passes independently on discovery, validation, and test;
2. positive patching recovery is significant on validation and held-out test (two-sided paired
   permutation test, Benjamini–Hochberg corrected `q < 0.05`);
3. mean test recovery is at least 0.20 and retains at least 50% of discovery recovery;
4. it exceeds the 95th percentile of its matched random-component null; and
5. its effect direction is positive in at least 75% of test pairs.

A candidate is rejected if any criterion fails, if its effect relies on a single prompt/template,
or if baseline eligibility fails. Reports must include rejected candidates, baseline failures,
excluded/degenerate pairs, null distributions, seeds, effect sizes, uncertainty, and error details;
an empty circuit is a valid result.

## Explicit non-goals

This MVP does not include Streamlit or any UI, Neuronpedia integration, multiple models, LLM-based
semantic labeling, open-ended “discover everything in a model” behavior, large-scale graph
visualization, or a full causal scanner. It establishes the reproducible environment and research
contract; dataset generation and baseline evaluation are the next implementation stage.
