import os
import torch
import streamlit as st

from main.devkota import Devkota
from src.core.config import ModelConfig
from src.core.train_spm import NepaliTokenizer
from src.core.utils import load_checkpoint


@st.cache_resource(show_spinner=True)
def load_tokenizer(path: str):
    return NepaliTokenizer(path)


@st.cache_resource(show_spinner=True)
def load_model(checkpoint_path: str):
    cfg = ModelConfig()
    model = Devkota(
        vocab_size=cfg.vocab_size,
        d_model=cfg.d_model,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        d_ff=cfg.d_ff,
        max_seq_len=cfg.max_seq_len,
        dropout=cfg.dropout,
        pad_idx=cfg.pad_idx,
    )

    if os.path.exists(checkpoint_path):
        load_checkpoint(checkpoint_path, model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    return model, device


def build_prompt(theme: str, lines: int) -> str:
    theme = theme.strip()
    if not theme:
        return "विषय: कविता\nकविता:"
    return f"विषय: {theme}\nकविता ({lines} पङ्क्ति):"


def generate_poem(model, device, tokenizer, prompt, max_tokens, temperature, top_k, top_p, repetition_penalty):
    input_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

    full = output_ids[0]
    new_tokens = full[input_ids.size(1):]
    if len(new_tokens) == 0:
        new_tokens = full
    return tokenizer.decode(new_tokens.tolist())


def main():
    st.title("Devkota POeLM")
    st.write("Nepali poem generation (fine-tune friendly).")

    ft_path = "checkpoints/finetuned/devkota_poet.pt"
    ckpt = ft_path if os.path.exists(ft_path) else "checkpoints/best_model.pt"

    tokenizer_path = "tokenizer/devkota_tokenizer.model"
    if not os.path.exists(tokenizer_path):
        st.error("Tokenizer missing. Run `python3 train_tokenizer.py` first.")
        return

    try:
        tokenizer = load_tokenizer(tokenizer_path)
    except Exception as e:
        st.error(f"Failed to load tokenizer: {e}")
        return

    if not os.path.exists(ckpt):
        st.warning("Checkpoint missing. Using randomly initialized model; outputs will be gibberish until you train.")
    model, device = load_model(ckpt)

    theme = st.text_input("थीम / Theme", "पहाड")
    lines = st.slider("Desired lines", min_value=2, max_value=10, value=4, step=1)
    max_tokens = st.slider("Max new tokens", min_value=20, max_value=200, value=80, step=10)
    temperature = st.slider("Temperature", min_value=0.2, max_value=1.2, value=0.7, step=0.05)
    top_k = st.slider("Top-k", min_value=5, max_value=200, value=40, step=5)
    top_p = st.slider("Top-p", min_value=0.5, max_value=1.0, value=0.85, step=0.01)
    repetition_penalty = st.slider("Repetition penalty", min_value=1.0, max_value=2.0, value=1.1, step=0.05)

    if st.button("Generate"):
        prompt = build_prompt(theme, lines)
        with st.spinner("Thinking..."):
            poem = generate_poem(
                model=model,
                device=device,
                tokenizer=tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
        st.subheader("Poem")
        st.write(poem)


if __name__ == "__main__":
    main()
