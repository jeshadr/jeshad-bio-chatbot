import json
import re
from pathlib import Path
import faiss, numpy as np
from openai import OpenAI

# ---------- Paths ----------
DATA_DIR = Path("data")
PROJECTS_DIR = DATA_DIR / "projects"
INDEX_DIR = Path("index")

# ---------- Model + style ----------
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
    "project list", "repos", "github projects", "github repos", "list of projects", "all of them", "all projects"
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

    # ---------- Index IO ----------
    def _load_meta(self, path: Path):
        raw = path.read_text(encoding="utf-8", errors="strict")
        obj = json.loads(raw)
        return obj["texts"], obj["meta"]

    # ---------- Embedding + Retrieval ----------
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

    # ---------- Modes ----------
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

    # ---------- Project parsing (deterministic) ----------
    def _read_text_file(self, p: Path) -> str:
        try:
            return p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return ""

    def _normalize(self, s: str) -> str:
        # normalize smart quotes and whitespace
        s = s.replace("—", "-").replace("–", "-").replace("’", "'").replace("“", '"').replace("”", '"')
        s = re.sub(r"\r\n?", "\n", s)
        return s

    def _scan_all_project_texts(self):
        texts = []
        if PROJECTS_DIR.exists():
            for p in sorted(PROJECTS_DIR.glob("*.txt")):
                t = self._read_text_file(p)
                if t:
                    texts.append(t)
        return texts

    def _extract_projects_from_text(self, text: str):
        text = self._normalize(text)
        items = []

        # One block per file, so just read keys
        m_name = re.search(r'^\s*Project\s*:\s*"?(.*?)"?\s*$', text, flags=re.IGNORECASE | re.MULTILINE)
        if not m_name:
            return items
        name = m_name.group(1).strip()

        m_desc = re.search(
            r'Description\s*:\s*(.+?)(?=\n\s*(Stack|What I Learned|What I learned|GitHub Link|Website Link)\s*:|\Z)',
            text, flags=re.IGNORECASE | re.DOTALL
        )
        desc = ""
        if m_desc:
            desc = re.sub(r"\s+", " ", m_desc.group(1).strip())
            parts = re.split(r"(?<=[.!?])\s+", desc)
            desc = " ".join(parts[:2]).strip()

        website = None
        github = None
        m_site = re.search(r'Website\s*Link\s*:\s*(https?://[^\s)>\]]+)', text, flags=re.IGNORECASE)
        if m_site:
            website = m_site.group(1).strip().rstrip(").,]")

        m_git = re.search(r'(GitHub\s*Link|GitHub)\s*:\s*(https?://[^\s)>\]]+)', text, flags=re.IGNORECASE)
        if m_git:
            github = m_git.group(2).strip().rstrip(").,]")

        items.append({"name": name, "desc": desc, "website": website, "github": github})
        return items

    def _extract_all_projects(self, hits=None, max_items=20):
        # Prefer deterministic file scan to avoid retrieval misses
        projects = []
        for t in self._scan_all_project_texts():
            projects.extend(self._extract_projects_from_text(t))

        # If nothing found in files, try to parse from retrieved hits as fallback
        if not projects and hits:
            for h in hits:
                projects.extend(self._extract_projects_from_text(h["text"]))

        # De-dupe by name
        seen = set()
        unique = []
        for p in projects:
            key = p["name"].lower()
            if key in seen:
                continue
            seen.add(key)
            unique.append(p)
        return unique[:max_items]

    def _render_projects_markdown(self, items):
        # Proper Markdown formatting
        lines = ["Here are some of my projects:", ""]
        for i, p in enumerate(items, 1):
            title = p["name"]
            desc = p["desc"]
            links = []
            if p["website"]:
                links.append(f"[Website]({p['website']})")
            if p["github"]:
                links.append(f"[GitHub]({p['github']})")
            link_str = " " + " · ".join(links) if links else ""
            lines.append(f"{i}. **{title}** - {desc}{link_str}")
        return "\n".join(lines)

    # ---------- General LLM plumbing ----------
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

    def _post_process(self, text: str, mode: str):
        text = text.replace("—", "-").strip()
        if mode != "list":
            parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()]
            if len(parts) > MAX_SENTENCES:
                text = " ".join(parts[:MAX_SENTENCES])
        return text

    # ---------- Public API ----------
    def answer(self, question: str, chat_history=None):
        mode = self._detect_mode(question)

        # Special case: list all projects deterministically from files
        if mode == "projects":
            items = self._extract_all_projects()
            if items:
                return self._render_projects_markdown(items)
            # If somehow no files were parsed, fall back to retrieval+LLM

        hits = self.retrieve(question, k=6)
        messages = self._build_messages(question, hits, chat_history, mode)
        resp = self.client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=TEMPERATURE,
        )
        out = resp.choices[0].message.content or ""
        return self._post_process(out, mode)
