# ingest.py
import os, re, json, sys
from pathlib import Path
from typing import List
from dotenv import load_dotenv

# Lightweight PDF text extraction
from pypdf import PdfReader
from openai import OpenAI
import faiss
import numpy as np

load_dotenv()

DATA_DIR = Path("data")
INDEX_DIR = Path("index")
INDEX_DIR.mkdir(exist_ok=True)

# Safety caps so you don't hang
PDF_MAX_PAGES = 25          # stop after N pages per PDF
MAX_CHARS_PER_FILE = 120_000  # skip anything larger
CHUNK_SIZE = 1400           # ~1k–1.5k chars works fine
CHUNK_OVERLAP = 220
EMBED_MODEL = "text-embedding-3-small"
BATCH_SIZE = 64

def log(msg: str):
    print(msg, flush=True)

def read_pdf_text(p: Path) -> str:
    try:
        reader = PdfReader(str(p))
        pages = min(len(reader.pages), PDF_MAX_PAGES)
        txt = []
        for i in range(pages):
            t = reader.pages[i].extract_text() or ""
            txt.append(t)
        if len(reader.pages) > PDF_MAX_PAGES:
            txt.append(f"\n[Truncated at {PDF_MAX_PAGES} pages]")
        return "\n".join(txt)
    except Exception as e:
        log(f"[WARN] Failed to read PDF {p.name}: {e}")
        return ""

def read_file(p: Path) -> str:
    if p.suffix.lower() == ".pdf":
        return read_pdf_text(p)
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        log(f"[WARN] Failed to read {p.name}: {e}")
        return ""

def normalize_whitespace(text: str) -> str:
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def chunk_chars(text: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> List[str]:
    # Paragraph-aware split, then pack
    paras = re.split(r"\n\s*\n", text)
    blocks = []
    cur = ""
    for para in paras:
        if len(cur) + len(para) + 2 <= size:
            cur += (("\n\n" if cur else "") + para)
        else:
            if cur:
                blocks.append(cur)
            # if single paragraph is huge, hard-split it
            if len(para) > size:
                for i in range(0, len(para), size - overlap):
                    blocks.append(para[i:i + size])
                cur = ""
            else:
                cur = para
    if cur:
        blocks.append(cur)

    # introduce overlaps between consecutive blocks
    if overlap and len(blocks) > 1:
        overlapped = []
        for i, b in enumerate(blocks):
            if i == 0:
                overlapped.append(b)
                continue
            prev = overlapped[-1]
            tail = prev[-overlap:] if len(prev) > overlap else prev
            merged = tail + "\n" + b
            overlapped[-1] = prev  # keep previous
            overlapped.append(merged[: size + overlap])
        return overlapped
    return blocks

def embed_batches(client: OpenAI, texts: List[str]) -> np.ndarray:
    vecs = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        log(f"Embedding {i+1}-{i+len(batch)} / {len(texts)}")
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        vecs.extend([d.embedding for d in resp.data])
    arr = np.array(vecs, dtype="float32")
    faiss.normalize_L2(arr)
    return arr

def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        log("[ERROR] OPENAI_API_KEY is not set.")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    corpus = []
    meta = []

    files = sorted([p for p in DATA_DIR.glob("*") if p.is_file()])
    if not files:
        log("[INFO] No files found in data/. Add resume.pdf, faq.txt, etc.")
        sys.exit(0)

    log(f"[INFO] Found {len(files)} file(s) in data/: {[p.name for p in files]}")

    for p in files:
        raw = read_file(p)
        raw = normalize_whitespace(raw)

        if not raw:
            log(f"[WARN] {p.name} produced no text. Skipping.")
            continue

        if len(raw) > MAX_CHARS_PER_FILE:
            log(f"[WARN] {p.name} exceeds {MAX_CHARS_PER_FILE} chars. Truncating.")
            raw = raw[:MAX_CHARS_PER_FILE] + "\n[Truncated]"

        chunks = chunk_chars(raw)
        log(f"[INFO] {p.name}: {len(chunks)} chunks")

        for i, ch in enumerate(chunks):
            corpus.append(ch)
            meta.append({"file": p.name, "chunk": i})

    if not corpus:
        log("[INFO] Nothing to embed.")
        sys.exit(0)

    log(f"[INFO] Total chunks: {len(corpus)}")
    vecs = embed_batches(client, corpus)

    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)

    faiss.write_index(index, str(INDEX_DIR / "bio.index"))
    (INDEX_DIR / "meta.json").write_text(
        json.dumps({"texts": corpus, "meta": meta}, ensure_ascii=False),
        encoding="utf-8",
    )

    log(f"[OK] Built index with {len(corpus)} chunks -> index/bio.index + index/meta.json")

if __name__ == "__main__":
    main()
