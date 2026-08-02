import torch
from transformer_lens import HookedTransformer


MODEL_NAME = "pythia-70m"
DEVICE = "cuda"

LAYER = 3
HEAD = 3

TESTS = [
    ("John", "apples", "Mary", "books"),
    ("David", "coffee", "Susan", "music"),
    ("Alice", "books", "Robert", "apples"),
    ("Michael", "music", "Sarah", "coffee"),
    ("Peter", "dogs", "Laura", "books"),
]


def single_token_id(model: HookedTransformer, text: str) -> int:
    tokens = model.to_tokens(text, prepend_bos=False)[0]

    if len(tokens) != 1:
        raise ValueError(
            f"{text!r} tek token değil: {model.to_str_tokens(tokens)}"
        )

    return int(tokens.item())


def ablate_head(activation: torch.Tensor, hook):
    modified = activation.clone()
    modified[:, :, HEAD, :] = 0
    return modified


def logit_difference(
    logits: torch.Tensor,
    target_id: int,
    distractor_id: int,
) -> float:
    final_logits = logits[0, -1].float()
    return float(final_logits[target_id] - final_logits[distractor_id])


def find_token_position(
    token_ids: torch.Tensor,
    searched_id: int,
) -> int:
    positions = (
        token_ids[0] == searched_id
    ).nonzero(as_tuple=False).flatten()

    if len(positions) == 0:
        raise ValueError(
            f"Token ID {searched_id} prompt içinde bulunamadı."
        )

    return int(positions[0].item())


def main() -> None:
    print(f"Loading {MODEL_NAME}...")

    model = HookedTransformer.from_pretrained(
        MODEL_NAME,
        device=DEVICE,
    )
    model.eval()

    print("Model loaded.")
    print(f"Testing Layer {LAYER} Head {HEAD}")
    print()

    results = []

    for subject, target, other_subject, distractor in TESTS:
        prompt = (
            f"{subject} likes {target}. "
            f"{other_subject} likes {distractor}. "
            f"{subject} likes"
        )

        target_text = f" {target}"
        distractor_text = f" {distractor}"

        try:
            target_id = single_token_id(model, target_text)
            distractor_id = single_token_id(model, distractor_text)
        except ValueError as error:
            print(f"SKIP: {error}")
            continue

        tokens = model.to_tokens(prompt)

        target_position = find_token_position(
            tokens,
            target_id,
        )

        distractor_position = find_token_position(
            tokens,
            distractor_id,
        )

        with torch.inference_mode():
            baseline_logits, cache = model.run_with_cache(tokens)

            ablated_logits = model.run_with_hooks(
                tokens,
                fwd_hooks=[
                    (
                        f"blocks.{LAYER}.attn.hook_z",
                        ablate_head,
                    )
                ],
            )

        baseline_score = logit_difference(
            baseline_logits,
            target_id,
            distractor_id,
        )

        ablated_score = logit_difference(
            ablated_logits,
            target_id,
            distractor_id,
        )

        drop = baseline_score - ablated_score

        pattern = cache[
            f"blocks.{LAYER}.attn.hook_pattern"
        ]

        final_attention = pattern[0, HEAD, -1]

        target_attention = float(
            final_attention[target_position]
        )

        distractor_attention = float(
            final_attention[distractor_position]
        )

        results.append(
            {
                "prompt": prompt,
                "baseline": baseline_score,
                "ablated": ablated_score,
                "drop": drop,
                "target_attention": target_attention,
                "distractor_attention": distractor_attention,
            }
        )

        print("=" * 90)
        print(f"Prompt: {prompt!r}")
        print(f"Target: {target_text!r}")
        print(f"Distractor: {distractor_text!r}")
        print(f"Baseline logit diff : {baseline_score:.4f}")
        print(f"Ablated logit diff  : {ablated_score:.4f}")
        print(f"Drop                : {drop:.4f}")
        print(f"Attention → target  : {target_attention:.4f}")
        print(f"Attention → distract: {distractor_attention:.4f}")

    if not results:
        print("Geçerli test kalmadı.")
        return

    mean_drop = sum(item["drop"] for item in results) / len(results)

    target_wins = sum(
        item["target_attention"] > item["distractor_attention"]
        for item in results
    )

    positive_drops = sum(
        item["drop"] > 0
        for item in results
    )

    print()
    print("=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(f"Valid tests                 : {len(results)}")
    print(f"Mean ablation drop          : {mean_drop:.4f}")
    print(
        "Positive causal effect      : "
        f"{positive_drops}/{len(results)}"
    )
    print(
        "Target attention > distractor: "
        f"{target_wins}/{len(results)}"
    )


if __name__ == "__main__":
    main()