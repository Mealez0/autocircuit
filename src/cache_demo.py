from transformer_lens import HookedTransformer

print("Loading model...")

model = HookedTransformer.from_pretrained(
    "pythia-70m",
    device="cuda"
)

print("Model loaded!")

prompt = "The capital of France is"

tokens = model.to_tokens(prompt)

logits, cache = model.run_with_cache(tokens)

print()
print("=" * 50)
print("Layers cached:", len(cache))
print("=" * 50)

for i, key in enumerate(cache.keys()):
    print(key)
    if i >= 20:
        break