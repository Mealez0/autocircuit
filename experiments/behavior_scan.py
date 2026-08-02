import torch
from transformer_lens import HookedTransformer


MODEL_NAME = "pythia-70m"
DEVICE = "cuda"
TOP_K = 10


TESTS = [
    {
        "category": "Factual recall",
        "prompt": "The capital of France is",
        "expected": " Paris",
    },
    {
        "category": "Number sequence",
        "prompt": "1 2 3 4 5",
        "expected": " 6",
    },
    {
        "category": "Letter sequence",
        "prompt": "A B C D",
        "expected": " E",
    },
    {
        "category": "Repetition",
        "prompt": "cat cat cat cat",
        "expected": " cat",
    },
    {
        "category": "Induction / copying",
        "prompt": "red blue green red blue",
        "expected": " green",
    },
    {
        "category": "Induction / copying",
        "prompt": "John likes apples. Mary likes oranges. John likes",
        "expected": " apples",
    },
    {
        "category": "Closing bracket",
        "prompt": "The result is (42",
        "expected": ")",
    },
    {
        "category": "Simple syntax",
        "prompt": "The opposite of hot is",
        "expected": " cold",
    },
]


def get_single_token_id(model: HookedTransformer, text: str) -> int | None:
    """Return token ID only if text is represented by exactly one token."""
    token_ids = model.to_tokens(text, prepend_bos=False)[0]

    if len(token_ids) != 1:
        return None

    return int(token_ids.item())


def analyze_test(
    model: HookedTransformer,
    category: str,
    prompt: str,
    expected: str,
) -> None:
    tokens = model.to_tokens(prompt)

    with torch.inference_mode():
        logits = model(tokens)

    final_logits = logits[0, -1]
    probabilities = torch.softmax(final_logits.float(), dim=-1)

    top_probabilities, top_token_ids = torch.topk(probabilities, TOP_K)

    print()
    print("=" * 80)
    print(f"Category : {category}")
    print(f"Prompt   : {prompt!r}")
    print(f"Tokens   : {model.to_str_tokens(tokens)}")
    print(f"Expected : {expected!r}")
    print("-" * 80)
    print(f"Top {TOP_K} predictions:")

    for rank, (token_id, probability) in enumerate(
        zip(top_token_ids.tolist(), top_probabilities.tolist()),
        start=1,
    ):
        token_text = model.to_string(token_id)
        print(
            f"{rank:>2}. {token_text!r:<20} "
            f"probability={probability:.6f}"
        )

    expected_token_id = get_single_token_id(model, expected)

    if expected_token_id is None:
        expected_tokens = model.to_str_tokens(
            model.to_tokens(expected, prepend_bos=False)
        )
        print()
        print(
            "Expected text is not a single token, so exact rank was not measured."
        )
        print(f"Expected tokenization: {expected_tokens}")
        return

    expected_probability = float(probabilities[expected_token_id])
    expected_logit = float(final_logits[expected_token_id])

    sorted_indices = torch.argsort(final_logits, descending=True)
    expected_rank = (
        torch.nonzero(sorted_indices == expected_token_id, as_tuple=False).item()
        + 1
    )

    print()
    print("Expected-token result:")
    print(f"Token       : {model.to_string(expected_token_id)!r}")
    print(f"Token ID    : {expected_token_id}")
    print(f"Rank        : {expected_rank}")
    print(f"Probability : {expected_probability:.8f}")
    print(f"Logit       : {expected_logit:.4f}")


def main() -> None:
    print(f"Loading {MODEL_NAME} on {DEVICE}...")

    model = HookedTransformer.from_pretrained(
        MODEL_NAME,
        device=DEVICE,
    )

    model.eval()

    print("Model loaded.")
    print(f"Layers: {model.cfg.n_layers}")
    print(f"Heads : {model.cfg.n_heads}")
    print(f"Width : {model.cfg.d_model}")

    for test in TESTS:
        analyze_test(
            model=model,
            category=test["category"],
            prompt=test["prompt"],
            expected=test["expected"],
        )

    print()
    print("=" * 80)
    print("Behavior scan complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()