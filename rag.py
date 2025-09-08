# rag.py
import json
import re
from pathlib import Path
import faiss, numpy as np
from openai import OpenAI

INDEX_DIR = Path("index")

# ===== Persona & style knobs =====
MODEL = "gpt-4o-mini"
TEMPERATURE = 0.35          # a touch warmer
MAX_SENTENCES = 4           # keep it tight for recruiters
ADD_FOLLOWUP = True         # only for project/role/skills/experience topics

SYSTEM_PROMPT = f"""
You are Jeshad speaking in first person. Sound human, warm, and direct.
Use plain, friendly language with short sentences. Never use em dashes.
Base answers only on the provided context. If the info is not in context, say so.
Keep answers under {MAX_SENTENCES} sentences.
You can use light humor sparingly. No slang that could confuse recruiters.
When helpful, ask one short question to keep the chat going, but only for project or role topics.
Never invent links or facts not in context. If asked for a link, only give it if it appears in the context.
"""

PERSONA = {
    "name": "Jeshad",
    "hello": "Hi! I'm Jeshad. Ask me anything about my projects, skills, or availability.",
    "how_are_you": "I'm doing well and always building. What can I help you with today?",
    "thanks": "You got it.",
    "project_followup": "Want a short summary of a project or a quick link to my resume from the context?",
}

# ===== intent regexes =====
_GREETING_RE   = re.compile(r"\b(hi|hello|hey|yo|sup|good (morning|afternoon|evening))\b", re.I)
_THANKS_RE     = re.compile(r"\b(thanks|thank you|appreciate it|ty)\b", re.I)
_HRU_RE        = re.compile(r"\b(how (are|r) (you|ya)|how's it going|how are things)\b", re.I)
_NAME_RE       = re.compile(r"\b(what('?s| is) your name|who are you)\b", re.I)
_SMALLTALK_RE  = re.compile(r"\b(joke|tell me a joke)\b", re.I)

# Topics that are allowed to trigger a follow-up question
_PROJECT_TOPICS = (
    re.compile(r"\b(project|projects|portfolio|repo|github|case study)\b", re.I),
    re.compile(r"\b(role|roles|responsibilit(y|ies)|what did you do)\b", re.I),
    re.compile(r"\b(skill|skills|tech(stack)?|stack|tools?)\b", re.I),
    re.compile(r"\b(experience|intern(ship)?|work history)\b", re.I),
)

def _strip_em_dashes(text: str) -> str:
    return text.replace("—", "-").replace("–", "-")

def _limit_sentences(text: str, max_sents: int) -> str:
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    if len(parts) <= max_sents:
        return text.strip()
    return " ".join(parts[:max_sents]).strip()

def _contains_link(text: str) -> bool:
    return bool(re.search(r"https?://", text))

def _detect_smalltalk(q: str):
    if _GREETING_RE.search(q):
        return "greeting"
    if _THANKS_RE.search(q):
        return "thanks"
    if _HRU_RE.search(q):
        return "how_are_you"
    if _NAME_RE.search(q):
        return "name"
    if _SMALLTALK_RE.search(q):
        return "smalltalk"
    return None

def _detect_topic(q: str) -> str | None:
    for rx in _PROJECT_TOPICS:
        if rx.search(q):
            return "projectish"
    return None

class BioRAG:
    def __init__(self):
        meta_path = INDEX_DIR / "meta.json"
        index_path = INDEX_DIR / "bio.index"
        if not meta_path.exists() or not index_path.exists():
            raise FileNotFoundError("Index not found. Run `python ingest.py` first.")

        self.client = OpenAI()
        self.texts, self.meta = self._load_meta(meta_path)
        self.index = faiss.read_index(str(index_path))

    def _load_meta(self, path: Path):
        raw = path.read_text(encoding="utf-8", errors="strict")
        obj = json.loads(raw)
        return obj["texts"], obj["meta"]

    def embed_query(self, q: str):
        vec = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=[q]
        ).data[0].embedding
        vec = np.array([vec], dtype="float32")
        faiss.normalize_L2(vec)
        return vec

    def retrieve(self, q: str, k=6):
        qv = self.embed_query(q)
        scores, idxs = self.index.search(qv, k)
        idxs = idxs[0].tolist()
        scs = scores[0].tolist()
        hits = []
        for i, s in zip(idxs, scs):
            if i >= 0:
                hits.append({"text": self.texts[i], "meta": self.meta[i], "score": s})
        return hits

    def _smalltalk_reply(self, kind: str):
        if kind == "greeting":
            return PERSONA["hello"]
        if kind == "thanks":
            return PERSONA["thanks"]
        if kind == "how_are_you":
            return PERSONA["how_are_you"]
        if kind == "name":
            return f"I'm {PERSONA['name']}. Ask me anything about my work."
        if kind == "smalltalk":
            return "I usually talk about projects, internships, and skills."
        return None

    def _build_messages(self, question: str, hits, chat_history):
        context = "\n\n".join(h["text"] for h in hits)
        user_content = f"Context:\n{context}\n\nQuestion:\n{question}"

        msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
        if chat_history:
            for m in chat_history[-8:]:
                msgs.append({"role": m["role"], "content": m["content"]})
        msgs.append({"role": "user", "content": user_content})
        return msgs, context

    def _post_process(self, text: str, context_text: str, topic: str | None):
        text = _strip_em_dashes(text)
        text = _limit_sentences(text, MAX_SENTENCES)

        # Remove links the context did not have
        if _contains_link(text) and not _contains_link(context_text):
            text = re.sub(r"https?://\S+", "[link unavailable in context]", text)

        # Only add a follow-up for projectish topics
        if ADD_FOLLOWUP and topic == "projectish":
            lower = text.lower().strip()
            if "?" not in text and not lower.startswith(("yes", "no")):
                text = text.rstrip() + " " + PERSONA["project_followup"]
        return text.strip()

    def answer(self, question: str, chat_history=None):
        q = question.strip()

        # Small talk path never adds a question
        kind = _detect_smalltalk(q)
        if kind:
            return self._smalltalk_reply(kind)

        # Topic detection for selective follow-ups
        topic = _detect_topic(q)

        # RAG flow
        hits = self.retrieve(q, k=6)
        messages, context_text = self._build_messages(q, hits, chat_history)

        resp = self.client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=TEMPERATURE,
        )
        out = resp.choices[0].message.content.strip()
        out = self._post_process(out, context_text, topic)

        # Clear nudge if context is missing
        if re.search(r"\b(not in context|no context|i don't have that|i do not have that)\b", out, re.I):
            out = "I don’t have that in my context yet. Ask about my projects, roles, tech stack, or availability."
        return out
