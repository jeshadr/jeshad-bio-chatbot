# app.py
import os
import sys
import subprocess
from pathlib import Path
import time

import streamlit as st
from dotenv import load_dotenv
from rag import BioRAG

# ---------- Config ----------
load_dotenv()  # local only; Streamlit Cloud uses st.secrets
st.set_page_config(page_title="Jeshad Bio Bot", page_icon="🗣️", layout="centered")

INDEX_DIR = Path("index")
BLOCK_TERMS = [
    "social security", "ssn", "home address", "salary history",
    "political affiliation", "bank account", "credit card",
]
GREETING = "Hey there! My name is Jeshad, you can ask me about my projects, experience or just talk with me!"

# Avatars
ASSISTANT_AVATAR = "assets/jeshad.png"  # commit this image
USER_AVATAR = "👤"

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
        if "OPENAI_API_KEY" not in env and "OPENAI_API_KEY" in st.secrets:
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

def header_with_reset():
    col1, col2 = st.columns([0.8, 0.2])
    with col1:
        st.title("Jeshad's Interactive Bio")
        st.caption("Ask about my experience, projects, skills, and availability.")
    with col2:
        if st.button("🔄 Reset chat", use_container_width=True):
            st.session_state.pop("chat", None)
            st.session_state.pop("rag", None)
            st.rerun()

# Small HTML/CSS typing indicator
TYPING_HTML = """
<style>
.typing { display:inline-flex; gap:6px; align-items:flex-end; height:1em; }
.typing .dot { width:6px; height:6px; border-radius:50%; background:#9aa4b2; animation:bounce 1s infinite ease-in-out; opacity:.85; }
.typing .dot:nth-child(2){ animation-delay:.15s; }
.typing .dot:nth-child(3){ animation-delay:.30s; }
@keyframes bounce {
  0%, 80%, 100% { transform: translateY(0); opacity:.5; }
  40% { transform: translateY(-6px); opacity:1; }
}
</style>
<div class="typing"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div>
"""

# ---------- UI ----------
header_with_reset()
ensure_index()

# Load RAG
try:
    if "rag" not in st.session_state:
        st.session_state.rag = BioRAG()
except Exception as e:
    st.error(f"Could not load the index: {e}")
    st.stop()

# Seed chat
if "chat" not in st.session_state:
    st.session_state.chat = [{"role": "assistant", "content": GREETING}]

# Render history (no keys needed)
for m in st.session_state.chat:
    if m["role"] == "assistant":
        with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
            st.markdown(m["content"])
    else:
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(m["content"])

# Input
q = st.chat_input("Ask me anything")
if q:
    # Render user bubble immediately
    st.session_state.chat.append({"role": "user", "content": q})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(q)

    if blocked(q):
        msg = "I do not share that. You can ask about my skills, projects, timeline, or availability."
        st.session_state.chat.append({"role": "assistant", "content": msg})
        with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
            st.markdown(msg)
    else:
        # Create a placeholder container for the assistant bubble
        placeholder = st.empty()
        with placeholder.container():
            with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
                st.markdown(TYPING_HTML, unsafe_allow_html=True)

        # Compute the answer
        a = st.session_state.rag.answer(q, chat_history=st.session_state.chat)

        # Naturalistic delay scaled by length (max 2.5s)
        delay = min(2.5, 0.015 * len(a))
        time.sleep(delay)

        # Replace the typing bubble with the final answer
        placeholder.empty()
        with placeholder.container():
            with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
                st.markdown(a)

        st.session_state.chat.append({"role": "assistant", "content": a})

# ---------- Disclaimer ----------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; font-size: 14px; color: gray;'>"
    "⚠️ Some information may be inaccurate. For the most up to date details, please visit my "
    "<a href='https://jeshadr.com' target='_blank'>portfolio at jeshadr.com</a>."
    "</div>",
    unsafe_allow_html=True,
)
