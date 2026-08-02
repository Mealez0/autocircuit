import torch
from transformer_lens import HookedTransformer


MODEL_NAME = "pythia-70m"
DEVICE = "cuda"

PROMPT = "John likes apples. Mary likes oranges. John likes"
LAYER = 3
HEAD = 3


def main() -> None:
    print(f"Loading {MODEL_NAME}...")

    model = HookedTransformer.from_pretrained(
        MODEL_NAME,
        device=DEVICE,
    )
    model.eval()

    tokens = model.to_tokens(PROMPT)

    with torch.inference_mode():
        _, cache = model.run_with_cache(tokens)

    token_strings = model.to_str_tokens(tokens)

    pattern = cache[f"blocks.{LAYER}.attn.hook_pattern"]

    # Shape: [batch, head, destination_position, source_position]
    head_pattern = pattern[0, HEAD]

    final_position_attention = head_pattern[-1]

    print()
    print("=" * 80)
    print(f"Prompt: {PROMPT!r}")
    print(f"Inspecting Layer {LAYER} Head {HEAD}")
    print("=" * 80)
    print()

    print("Token positions:")
    for index, token in enumerate(token_strings):
        print(f"{index:>2}: {token!r}")

    print()
    print("Attention from the FINAL token position:")
    print()

    ranked = sorted(
        enumerate(final_position_attention.tolist()),
        key=lambda item: item[1],
        reverse=True,
    )

    for source_position, weight in ranked:
        print(
            f"source={source_position:>2} "
            f"token={token_strings[source_position]!r:<18} "
            f"attention={weight:.6f}"
        )

    print()
    print("=" * 80)
    print("Full attention matrix")
    print("=" * 80)

    for destination_position, row in enumerate(head_pattern.tolist()):
        print()
        print(
            f"Destination {destination_position}: "
            f"{token_strings[destination_position]!r}"
        )

        ranked_row = sorted(
            enumerate(row),
            key=lambda item: item[1],
            reverse=True,
        )

        for source_position, weight in ranked_row[:5]:
            print(
                f"  -> source={source_position:>2} "
                f"token={token_strings[source_position]!r:<18} "
                f"attention={weight:.6f}"
            )


if __name__ == "__main__":
    main()