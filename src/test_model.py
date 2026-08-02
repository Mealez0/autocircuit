from transformer_lens import HookedTransformer

print("Loading model...")

model = HookedTransformer.from_pretrained(
    "pythia-70m",
    device="cuda"
)

print("Model loaded!")

prompt = "The capital of France is"

tokens = model.to_tokens(prompt)

logits = model(tokens)

print("Logits shape:", logits.shape)