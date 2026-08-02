import torch
from transformer_lens import HookedTransformer


MODEL_NAME = "pythia-70m"
DEVICE = "cuda"

TESTS = [
    (
        "Clean",
        "John likes apples. Mary likes oranges. John likes",
        " apples",
    ),
    (
        "Swapped objects",
        "John likes oranges. Mary likes apples. John likes",
        " oranges",
    ),
    (
        "Changed person",
        "David likes apples. Mary likes oranges. David likes",
        " apples",
    ),
    (
        "Corrupted first object",
        "John likes books. Mary likes oranges. John likes",
        " books",
    ),
    (
        "No matching history",
        "Mary likes oranges. Peter likes books. John likes",
        " apples",
    ),
]


def score_completion(
    model: HookedTransformer,
    prompt: str,
    completion: str,
) -> dict:
    prompt_tokens = model.to_tokens(prompt)
    completion_tokens = model.to_tokens(
        completion,
        prepend_bos=False,
    )[0]

    current_tokens = prompt_tokens
    token_results = []
    total_log_probability = 0.0

    with torch.inference_mode():
        for token_id in completion_tokens:
            logits = model(current_tokens)[0, -1].float()
            log_probabilities = torch.log_softmax(logits, dim=-1)
            probabilities = torch.softmax(logits, dim=-1)

            token_id_int = int(token_id.item())
            probability = float(probabilities[token_id_int])
            log_probability = float(log_probabilities[token_id_int])

            sorted_ids = torch.argsort(logits, descending=True)
            rank = int(
                torch.nonzero(
                    sorted_ids == token_id_int,
                    as_tuple=False,
                ).item()
            ) + 1

            token_results.append(
                {
                    "token": model.to_string(token_id_int),
                    "token_id": token_id_int,
                    "probability": probability,
                    "log_probability": log_probability,
                    "rank": rank,
                }
            )

            total_log_probability += log_probability

            current_tokens = torch.cat(
                [
                    current_tokens,
                    token_id.view(1, 1).to(current_tokens.device),
                ],
                dim=1,
            )

    return {
        "tokens": token_results,
        "total_log_probability": total_log_probability,
        "average_log_probability": (
            total_log_probability / len(token_results)
        ),
    }


def get_top_predictions(
    model: HookedTransformer,
    prompt: str,
    top_k: int = 5,
) -> list[tuple[str, float]]:
    tokens = model.to_tokens(prompt)

    with torch.inference_mode():
        logits = model(tokens)[0, -1].float()
        probabilities = torch.softmax(logits, dim=-1)

    values, ids = torch.topk(probabilities, top_k)

    return [
        (model.to_string(token_id), float(probability))
        for token_id, probability in zip(
            ids.tolist(),
            values.tolist(),
        )
    ]


def main() -> None:
    print(f"Loading {MODEL_NAME}...")

    model = HookedTransformer.from_pretrained(
        MODEL_NAME,
        device=DEVICE,
    )
    model.eval()

    print("Model loaded.")
    print()

    for name, prompt, expected in TESTS:
        result = score_completion(
            model=model,
            prompt=prompt,
            completion=expected,
        )

        top_predictions = get_top_predictions(
            model=model,
            prompt=prompt,
        )

        print("=" * 80)
        print(f"Test     : {name}")
        print(f"Prompt   : {prompt!r}")
        print(f"Expected : {expected!r}")
        print(
            "Tokens   :",
            [item["token"] for item in result["tokens"]],
        )
        print("Top 5 first-token predictions:")

        for rank, (token, probability) in enumerate(
            top_predictions,
            start=1,
        ):
            print(
                f"  {rank}. {token!r:<18} "
                f"{probability:.6f}"
            )

        print("Expected completion details:")

        for index, item in enumerate(
            result["tokens"],
            start=1,
        ):
            print(
                f"  Token {index}: {item['token']!r:<12} "
                f"rank={item['rank']:<5} "
                f"prob={item['probability']:.6f}"
            )

        print(
            "Average log probability:",
            f"{result['average_log_probability']:.6f}",
        )
        print()

    print("=" * 80)
    print("Association test complete.")


if __name__ == "__main__":
    main()