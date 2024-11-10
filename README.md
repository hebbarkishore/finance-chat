# Finance Chat — Offline Finance Chatbot

A 100% local, free chatbot for finance questions. Uses `google/flan-t5-base` with a small TF-IDF glossary retriever. No API keys.

## Quickstart
```bash
python3 -m venv .venv 
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py