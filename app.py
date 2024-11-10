import json
import os
from typing import List, Tuple

import streamlit as st
import numpy as np

# Lightweight retrieval
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Local LLM (instruction-tuned, small, CPU friendly)
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# -----------------------------
# Config
# -----------------------------
MODEL_NAME = "google/flan-t5-base"  # ~77M params, good for CPU
MAX_INPUT_TOKENS = 512
MAX_NEW_TOKENS = 256

st.set_page_config(page_title="Finance Q&A", layout="centered")
st.title("💬 FinChat-Local — Offline Finance Chatbot")
st.caption("Runs 100% locally on CPU. No API keys. Ask finance questions (e.g., APR, ETF vs mutual fund, compound interest).")


# -----------------------------
# Utilities
# -----------------------------
@st.cache_resource(show_spinner=True)
def load_model_and_tokenizer(model_name: str):
    """
    Downloads on first run, then cached locally.
    """
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tok, mdl


@st.cache_resource(show_spinner=False)
def load_glossary(path: str) -> List[Tuple[str, str]]:
    """
    Loads a small finance glossary: list of (term, definition).
    """
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Accept both {"term": "definition", ...} or [{"term": "...","definition":"..."}]
    if isinstance(data, dict):
        items = list(data.items())
    else:
        items = [(d.get("term", ""), d.get("definition", "")) for d in data]
    # Basic cleanup
    items = [(t.strip(), d.strip()) for (t, d) in items if t and d]
    return items


def build_retriever(pairs: List[Tuple[str, str]]):
    """
    Builds a simple TF-IDF retriever over terms+definitions.
    """
    if not pairs:
        return None, None, None
    docs = [f"{t}. {d}" for (t, d) in pairs]
    vec = TfidfVectorizer(stop_words="english", lowercase=True)
    X = vec.fit_transform(docs)
    return vec, X, docs


def retrieve_similar(query, vec, X, docs, top_k=3, threshold=0.4):
    if vec is None:
        return []
    qv = vec.transform([query])
    sims = cosine_similarity(qv, X)[0]
    idxs = np.argsort(sims)[::-1][:top_k]
    results = [(float(sims[i]), docs[i]) for i in idxs if sims[i] > threshold]
    return results


def make_prompt(question: str, retrieved_notes: list[str]) -> str:
    context = ""
    if retrieved_notes:
        context = "\n".join(f"- {n}" for n in retrieved_notes)
    return (
        "You are a certified financial advisor who explains clearly with examples.\n"
        "Use the context below only if it helps; otherwise answer from your own knowledge.\n"
        f"{context}\n\n"
        f"Question: {question}\n"
        "Answer in 3–5 concise sentences suitable for a beginner:\n"
    )


def generate_answer(tokenizer, model, prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_INPUT_TOKENS)
    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        top_p=0.9,
        temperature=0.8,
        num_beams=1,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


# -----------------------------
# Load resources
# -----------------------------
with st.spinner("Loading model (first run downloads locally)…"):
    tokenizer, model = load_model_and_tokenizer(MODEL_NAME)

glossary_pairs = load_glossary("data/finance_glossary.json")
vec, X, docs = build_retriever(glossary_pairs)

# Session state for chat
if "history" not in st.session_state:
    st.session_state.history = [
        {"role": "assistant", "content": "Hi! I’m a local, offline finance bot. Ask me anything about APR, ETFs, compounding, etc."}
    ]

# Show chat so far
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input
user_input = st.chat_input("Type your finance question…")
if user_input:
    # Show user msg
    st.session_state.history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Retrieve relevant notes from glossary (optional)
    retrieved = retrieve_similar(user_input, vec, X, docs, top_k=3)
    retrieved_notes = [note for (_score, note) in retrieved]

    # Build prompt and generate
    prompt = make_prompt(user_input, retrieved_notes)
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            answer = generate_answer(tokenizer, model, prompt)
            st.markdown(answer)
            # Show sources if any
            if retrieved_notes:
                with st.expander("Sources / context (local glossary)"):
                    for s, note in retrieved[:3]:
                        st.write(f"• ({s:.2f}) {note}")

    # Save assistant reply
    st.session_state.history.append({"role": "assistant", "content": answer})

# Download chat button
st.download_button(
    "⬇️ Download Chat (JSON)",
    data=json.dumps(st.session_state.history, ensure_ascii=False, indent=2),
    file_name="finchat_local_chat.json",
    mime="application/json",
)