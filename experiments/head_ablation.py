import torch
from transformer_lens import HookedTransformer


MODEL_NAME = "pythia-70m"
DEVICE = "cuda"

PROMPT = "John likes apples. Mary likes oranges. John likes"
TARGET = " apples"
DISTRACTOR = " books"


def get_single_token_id(model: HookedTransformer, text: str) -> int:
    tokens = model.to_tokens(text, prepend_bos=False)[0]

    if len(tokens) != 1:
        raise ValueError(
            f"{text!r} tek token değil: {model.to_str_tokens(tokens)}"
        )

    return int(tokens.item())


def get_logit_difference(
    model: HookedTransformer,
    tokens: torch.Tensor,
    target_id: int,
    distractor_id: int,
    hooks=None,
) -> float:
    with torch.inference_mode():
        if hooks is None:
            logits = model(tokens)
        else:
            logits = model.run_with_hooks(
                tokens,
                fwd_hooks=hooks,
            )

    final_logits = logits[0, -1].float()

    return float(
        final_logits[target_id] - final_logits[distractor_id]
    )


def make_head_ablation_hook(head_index: int):
    def ablate_head(activation: torch.Tensor, hook):
        modified = activation.clone()

        # hook_z biçimi:
        # [batch, position, head, d_head]
        modified[:, :, head_index, :] = 0

        return modified

    return ablate_head


def main() -> None:
    print(f"Loading {MODEL_NAME}...")

    model = HookedTransformer.from_pretrained(
        MODEL_NAME,
        device=DEVICE,
    )
    model.eval()

    tokens = model.to_tokens(PROMPT)

    target_id = get_single_token_id(model, TARGET)
    distractor_id = get_single_token_id(model, DISTRACTOR)

    print("Model loaded.")
    print(f"Prompt     : {PROMPT!r}")
    print(f"Target     : {TARGET!r}")
    print(f"Distractor : {DISTRACTOR!r}")
    print(f"Layers     : {model.cfg.n_layers}")
    print(f"Heads      : {model.cfg.n_heads}")
    print()

    baseline = get_logit_difference(
        model=model,
        tokens=tokens,
        target_id=target_id,
        distractor_id=distractor_id,
    )

    print(f"Baseline target-distractor logit difference: {baseline:.6f}")
    print()

    results = []

    for layer in range(model.cfg.n_layers):
        hook_name = f"blocks.{layer}.attn.hook_z"

        for head in range(model.cfg.n_heads):
            ablated_score = get_logit_difference(
                model=model,
                tokens=tokens,
                target_id=target_id,
                distractor_id=distractor_id,
                hooks=[
                    (
                        hook_name,
                        make_head_ablation_hook(head),
                    )
                ],
            )

            drop = baseline - ablated_score

            results.append(
                {
                    "layer": layer,
                    "head": head,
                    "ablated_score": ablated_score,
                    "drop": drop,
                }
            )

            print(
                f"Layer {layer} Head {head} | "
                f"score={ablated_score:>9.4f} | "
                f"drop={drop:>9.4f}"
            )

    results.sort(
        key=lambda item: item["drop"],
        reverse=True,
    )

    print()
    print("=" * 80)
    print("TOP 15 MOST IMPORTANT HEADS")
    print("=" * 80)

    for rank, item in enumerate(results[:15], start=1):
        print(
            f"{rank:>2}. "
            f"Layer {item['layer']} Head {item['head']} | "
            f"drop={item['drop']:.6f} | "
            f"ablated_score={item['ablated_score']:.6f}"
        )

    print()
    print("=" * 80)
    print("Heads with a positive drop helped predict the target.")
    print("A large positive drop means ablating that head damaged the behavior.")
    print("=" * 80)


if __name__ == "__main__":
    main()