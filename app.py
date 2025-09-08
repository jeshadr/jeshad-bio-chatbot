# app.py
import os
import sys
import subprocess
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from rag import BioRAG

# ---------- Config ----------
load_dotenv()  # works locally; Cloud will use st.secrets
st.set_page_config(page_title="Jeshad Bio Bot", page_icon="🗣️", layout="centered")

INDEX_DIR = Path("index")
BLOCK_TERMS = [
    "social security", "ssn", "home address", "salary history",
    "political affiliation", "bank account", "credit card"
]

# ---------- Helpers ----------
def ensure_index():
    """Build the FAISS index on first run if missing."""
    meta = INDEX_DIR / "meta.json"
    bio = INDEX_DIR / "bio.index"
    if meta.exists() and bio.exists():
        return

    INDEX_DIR.mkdir(exist_ok=True)

    with st.spinner("Building index for the first run..."):
        env = os.environ.copy()
        # On Streamlit Cloud, prefer Secrets
        if "OPENAI_API_KEY" not in env:
            if "OPENAI_API_KEY" in st.secrets:
                env["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
        if not env.get("OPENAI_API_KEY"):
            st.error("OPENAI_API_KEY is not set. Add it in Streamlit Secrets.")
            st.stop()

        proc = subprocess.run(
            [sys.executable, "ingest.py"],
            env=env,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            st.error("Index build failed.")
            if proc.stdout:
                st.write("Stdout:")
                st.code(proc.stdout)
            if proc.stderr:
                st.write("Stderr:")
                st.code(proc.stderr)
            st.stop()

def blocked(q: str) -> bool:
    ql = q.lower()
    return any(term in ql for term in BLOCK_TERMS)

def print_header():
    st.title("Jeshad's Interactive Bio")
    st.caption("Ask about my experience, projects, skills, and availability.")

# ---------- UI ----------
print_header()

with st.sidebar:
    st.subheader("Controls")
    if st.button("Reset chat"):
        st.session_state.pop("chat", None)
        st.session_state.pop("rag", None)
        st.experimental_rerun()
    st.markdown("Secrets are read from **Streamlit Secrets** in the cloud.")

# Make sure index exists on this machine
ensure_index()

# Load RAG
try:
    if "rag" not in st.session_state:
        st.session_state.rag = BioRAG()
except Exception as e:
    st.error(f"Could not load the index: {e}")
    st.stop()

# Session chat history
if "chat" not in st.session_state:
    st.session_state.chat = []

# Render history
for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Input
q = st.chat_input("Ask me anything")
if q:
    # Basic guardrails
    if blocked(q):
        msg = "I do not share that. You can ask about my skills, projects, timeline, or availability."
        st.session_state.chat.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            st.markdown(msg)
        st.session_state.chat.append({"role": "assistant", "content": msg})
    else:
        st.session_state.chat.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.markdown(q)

        with st.chat_message("assistant"):
            a = st.session_state.rag.answer(q, chat_history=st.session_state.chat)
            st.markdown(a)
            st.session_state.chat.append({"role": "assistant", "content": a})
