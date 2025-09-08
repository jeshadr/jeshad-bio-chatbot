# rag.py
import json
from pathlib import Path
import faiss, numpy as np
from openai import OpenAI

INDEX_DIR = Path("index")

# ===== Persona & style knobs =====
MODEL = "gpt-4o-mini"
TEMPERATURE = 0.3           # a touch more warmth than 0.2
MAX_SENTENCES = 6           # keep it tight for recruiters
ADD_FOLLOWUP = True         # ask a short follow-up when it helps

SYSTEM_PROMPT = f"""
You are Jeshad speaking in first person. Be warm, open, and professional.
Use plain, friendly language with short sentences. No em dashes.
Base answers only on the provided context. If the info is not in context, say so clearly.
Keep answers under {MAX_SENTENCES} sentences.
When it helps, end with one brief question to keep the conversation going.
Never invent links or facts not in context. If asked for a link, only provide it if it appears in context.
"""

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
        hits = []
        for i in idxs:
            if i >= 0:
                hits.append({"text": self.texts[i], "meta": self.meta[i]})
        return hits

    def _build_messages(self, question: str, hits, chat_history):
        context = "\n\n".join(h["text"] for h in hits)
        user_content = f"Context:\n{context}\n\nQuestion:\n{question}"

        msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
        if chat_history:
            # keep prior turns to preserve vibe, but keep it lean
            for m in chat_history[-8:]:
                msgs.append({"role": m["role"], "content": m["content"]})
        msgs.append({"role": "user", "content": user_content})
        return msgs

    def _post_process(self, text: str):
        # Hard stop on em dashes just in case the model slips one in
        text = text.replace("—", "-")
        if ADD_FOLLOWUP:
            # If there is no question mark and it isn't a yes/no or link-only reply, add a gentle follow-up.
            lower = text.lower()
            if "?" not in text and not lower.strip().startswith(("yes", "no")):
                text = text.strip() + " Would you like a quick link to my resume or a short summary of a project?"
        return text

    def answer(self, question: str, chat_history=None):
        hits = self.retrieve(question, k=6)
        messages = self._build_messages(question, hits, chat_history)

        resp = self.client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=TEMPERATURE,
        )
        out = resp.choices[0].message.content.strip()
        out = self._post_process(out)
        return out
