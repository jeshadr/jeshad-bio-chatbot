import json
import re
from pathlib import Path
import faiss, numpy as np
from openai import OpenAI

INDEX_DIR = Path("index")

MODEL = "gpt-4o-mini"
TEMPERATURE = 0.6
MAX_SENTENCES = 6
ENABLE_LIST_MODE = True
ADD_FOLLOWUP = False

VOICE_GUIDE = f"""
You are Jeshad, speaking in first person. Friendly, confident, approachable.
Use plain words, short sentences, and contractions. No em dashes. Avoid buzzwords.
Answer ONLY from the provided context and the app description. If info is missing, say so.
Keep answers under {MAX_SENTENCES} sentences unless the user clearly wants a list.
Do not invent links. Only use links that appear in the provided context.
Prefer direct answers. Do not describe the dataset unless asked.
"""

APP_CONTEXT = """
This is my interactive bio chatbot. Ask me about my projects, experience, skills, and availability.
I answer in my voice using facts from my files. I do not share private data.
"""

SMALL_TALK = {"hi", "hello", "hey", "yo", "how are you", "what's up", "whats up"}

BOT_META_TRIGGERS = [
    "what is this", "who are you", "who am i talking to", "what can you do",
    "how does this work", "what is this app", "what is this bot", "purpose",
    "capabilities", "privacy"
]

PROJECT_TRIGGERS = [
    "projects", "show projects", "your projects", "portfolio projects",
    "project list", "repos", "github projects", "github repos"
]

URL_RE = re.compile(r"https?://[^\s)>\]]+", re.IGNORECASE)

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

    def retrieve(self, q: str, k=4):
        qv = self.embed_query(q)
        scores, idxs = self.index.search(qv, k)
        idxs = idxs[0].tolist()
        hits = []
        for i in idxs:
            if i >= 0:
                hits.append({"text": self.texts[i], "meta": self.meta[i]})
        return hits

    def _detect_mode(self, question: str):
        q = question.lower().strip()
        if q in SMALL_TALK or "how are you" in q:
            return "small_talk"
        if any(p in q for p in ["tell me about yourself", "about yourself", "overview", "summary", "intro"]):
            return "self_intro"
        if any(t in q for t in BOT_META_TRIGGERS):
            return "bot_meta"
        if any(t in q for t in PROJECT_TRIGGERS):
            return "projects"
        if ENABLE_LIST_MODE and (any(p in q for p in ["list", "top", "bullet", "bullets"]) or re.search(r"\bskills?\b|\bprojects?\b", q)):
            return "list"
        return "default"

    def _few_shots(self):
        return [
            {"role": "user", "content": "Question:\nWhat is this?\nContext:\n"},
            {"role": "assistant", "content": "This is my interactive bio. Ask about my projects, stack, or availability and I’ll answer in my voice using facts from my files."},
        ]

    def _gather_links(self, hits):
        links = []
        for h in hits:
            for m in URL_RE.findall(h["text"]):
                links.append(m.strip().rstrip(").,]"))
        # de-dupe while preserving order
        seen = set()
        out = []
        for u in links:
            if u not in seen:
                seen.add(u)
                out.append(u)
        return out

    def _build_messages(self, question: str, hits, chat_history, mode: str):
        base_context = f"App:\n{APP_CONTEXT.strip()}\n"
        docs_context = "\n\n".join(h["text"] for h in hits)
        context = base_context + ("\nSources:\n" + docs_context if docs_context else "")

        style_hint = ""
        if mode == "self_intro":
            style_hint = "Start with one friendly hook sentence, then a short paragraph, then up to 3 quick bullets."
        elif mode == "list":
            style_hint = "Return a short numbered list, max 5 items, one line per item."
        elif mode == "small_talk":
            style_hint = "Greet briefly, then invite a job-related question."
        elif mode == "bot_meta":
            style_hint = "Explain what this app is and what the user can ask. Keep it short and helpful."
        elif mode == "projects":
            style_hint = "Return a short list of projects and include a working link for each one that appears in the context. Do not invent links."

        msgs = [{"role": "system", "content": VOICE_GUIDE}]
        msgs.extend(self._few_shots())
        if chat_history:
            for m in chat_history[-8:]:
                msgs.append({"role": m["role"], "content": m["content"]})
        msgs.append({
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion:\n{question}\n\nStyle hint: {style_hint}"
        })
        return msgs

    def _post_process(self, text: str, mode: str, links_from_hits=None):
        text = text.replace("—", "-").strip()

        # If user asked for projects, ensure at least one link is present if we have them
        if mode == "projects" and links_from_hits:
            has_link = "http://" in text or "https://" in text
            if not has_link:
                # Append a compact links section
                top = links_from_hits[:5]
                lines = "\n".join(f"- {u}" for u in top)
                text = f"{text}\n\nLinks:\n{lines}"

        # Keep responses tight unless list mode
        if mode not in {"list", "projects"}:
            parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()]
            if len(parts) > MAX_SENTENCES:
                text = " ".join(parts[:MAX_SENTENCES])
        return text

    def answer(self, question: str, chat_history=None):
        mode = self._detect_mode(question)
        hits = self.retrieve(question, k=4)
        links = self._gather_links(hits)
        messages = self._build_messages(question, hits, chat_history, mode)

        # Bias the model by showing links when relevant via the user content context
        if mode == "projects" and links:
            messages.append({
                "role": "user",
                "content": "These links were found in the sources. Use them directly if you reference the corresponding projects:\n" + "\n".join(links[:8])
            })

        resp = self.client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=TEMPERATURE,
        )
        out = resp.choices[0].message.content or ""
        return self._post_process(out, mode, links_from_hits=links)
