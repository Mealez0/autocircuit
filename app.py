import streamlit as st
from transformer_lens import HookedTransformer

st.set_page_config(
    page_title="Mechanistic Interpretability Studio",
    layout="wide"
)

st.title("🧠 Mechanistic Interpretability Studio")

# -------------------------------
# Model sadece bir kez yüklensin
# -------------------------------

@st.cache_resource
def load_model():
    return HookedTransformer.from_pretrained(
        "pythia-70m",
        device="cuda"
    )

model = load_model()

# -------------------------------

prompt = st.text_input(
    "Prompt",
    "The capital of France is"
)

if st.button("🚀 Run"):

    with st.spinner("Running Pythia..."):

        tokens = model.to_tokens(prompt)
        logits, cache = model.run_with_cache(tokens)

    st.success("Inference Complete!")

    col1, col2 = st.columns([1,2])

    with col1:

        st.subheader("📚 Hook Points")

        hook = st.selectbox(
            "Select Hook",
            list(cache.keys())
        )

    with col2:

        tensor = cache[hook]

        st.subheader(hook)

        st.write("### Shape")
        st.code(str(tuple(tensor.shape)))

        st.write("### Statistics")

        c1,c2,c3,c4 = st.columns(4)

        c1.metric("Mean", f"{tensor.float().mean():.5f}")
        c2.metric("Std", f"{tensor.float().std():.5f}")
        c3.metric("Min", f"{tensor.float().min():.5f}")
        c4.metric("Max", f"{tensor.float().max():.5f}")

        st.write("### Tensor")

        st.write(tensor)