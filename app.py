import os
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

st.set_page_config(page_title="Balochi GPT", page_icon="📝", layout="centered")
st.title("Balochi GPT Text Generation")
st.write("Generate a continuation in Eastern Balochi Latin script.")

try:
    MODEL_ID = st.secrets["HF_MODEL_ID"]
except Exception:
    MODEL_ID = os.getenv("HF_MODEL_ID", "sajida12/balochi-gpt2-generation")

@st.cache_resource
def load_generator():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model

prompt = st.text_area("Balochi prompt", placeholder="Man Balochi zaban...")
max_new = st.slider("New tokens", 20, 120, 60)
temperature = st.slider("Temperature", 0.2, 1.2, 0.8, 0.1)
top_p = st.slider("Top-p", 0.1, 1.0, 0.9, 0.05)

if st.button("Generate text", type="primary"):
    if not prompt.strip():
        st.warning("Please enter a Balochi prompt.")
    elif "YOUR_HF_USERNAME" in MODEL_ID:
        st.error("Set HF_MODEL_ID to your uploaded Hugging Face model, or edit MODEL_ID in app.py.")
    else:
        try:
            tokenizer, model = load_generator()
            # Goldfish model card recommends prepending its [CLS] (= [BOS]) token.
            model_input = prompt.strip()
            if tokenizer.cls_token:
                model_input = tokenizer.cls_token + model_input
            inputs = tokenizer(model_input, return_tensors="pt", truncation=True,
                               max_length=min(400, getattr(model.config, "n_positions", 512)))
            with torch.no_grad():
                output = model.generate(
                    **inputs, max_new_tokens=max_new, do_sample=True,
                    temperature=temperature, top_p=top_p,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            generated = tokenizer.decode(output[0], skip_special_tokens=True)
            st.subheader("Generated Balochi text")
            st.write(generated)
            st.caption("Generated text may contain grammatical or cultural errors. Ask a fluent Balochi speaker to review it.")
        except Exception as exc:
            st.error(f"Model loading or generation failed: {exc}")
