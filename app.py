import streamlit as st
from dotenv import load_dotenv
from rag import BioRAG

load_dotenv()
st.set_page_config(page_title="Jeshad Bio Bot", page_icon="🗣️", layout="centered")

st.title("Jeshad's Interactive Bio")
st.caption("Ask about my experience, projects, skills, and availability.")

if "rag" not in st.session_state:
    st.session_state.rag = BioRAG()
if "chat" not in st.session_state:
    st.session_state.chat = []

for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

q = st.chat_input("Ask me anything")
if q:
    st.session_state.chat.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)

    with st.chat_message("assistant"):
        a = st.session_state.rag.answer(q, chat_history=st.session_state.chat)
        st.markdown(a)
        st.session_state.chat.append({"role": "assistant", "content": a})
