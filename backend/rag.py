"""
Minimal RAG pipeline: ingest .txt/.md files from KNOWLEDGE_DIR into a local
Chroma vector store, and retrieve the top-k most relevant chunks for a query.
Embeddings run locally (chroma's built-in all-MiniLM-L6-v2 model, downloaded
once on first run) -> no API cost.
"""
import glob
import logging
import os

import chromadb

import config

logger = logging.getLogger("voice-agent.rag")

_client = None
_collection = None


def _get_collection():
    global _client, _collection
    if _collection is None:
        os.makedirs(config.CHROMA_PATH, exist_ok=True)
        _client = chromadb.PersistentClient(path=config.CHROMA_PATH)
        # no embedding_function given -> chroma uses its built-in local one
        _collection = _client.get_or_create_collection(name="knowledge_base")
    return _collection


def _chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """Simple sliding-window chunking by characters (good enough for a demo KB)."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return [c.strip() for c in chunks if c.strip()]


def ingest_knowledge_base(knowledge_dir: str = config.KNOWLEDGE_DIR) -> int:
    """
    Read every .txt/.md file in knowledge_dir, chunk it, and put it in Chroma.
    Returns the number of chunks ingested.

    The folder is the source of truth, so anything already in the store is
    cleared first. Otherwise chunks from files you edited or deleted would
    stay around and the assistant would keep quoting them.
    """
    collection = _get_collection()

    old_ids = collection.get()["ids"]
    if old_ids:
        collection.delete(ids=old_ids)

    files = glob.glob(os.path.join(knowledge_dir, "**/*.txt"), recursive=True) + \
            glob.glob(os.path.join(knowledge_dir, "**/*.md"), recursive=True)

    total = 0
    for path in files:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        chunks = _chunk_text(text)
        if not chunks:
            continue
        # relative path (not just the file name) so two files called
        # notes.txt in different folders don't overwrite each other
        name = os.path.relpath(path, knowledge_dir)
        ids = [f"{name}::{i}" for i in range(len(chunks))]
        metadatas = [{"source": name} for _ in chunks]
        collection.upsert(documents=chunks, ids=ids, metadatas=metadatas)
        total += len(chunks)
        logger.info(f"Ingested {len(chunks)} chunks from {path}")

    return total


def retrieve(query: str, top_k: int = config.RAG_TOP_K) -> list[dict]:
    """
    Return the top_k most relevant chunks for `query`, each as
    {"text": ..., "source": ..., "distance": ...}. Empty list if the KB
    is empty or nothing is close enough to be useful.
    """
    collection = _get_collection()
    if collection.count() == 0:
        return []

    results = collection.query(query_texts=[query], n_results=min(top_k, collection.count()))

    hits = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]
    for doc, meta, dist in zip(docs, metas, dists):
        hits.append({"text": doc, "source": meta.get("source", "unknown"), "distance": dist})
    return hits


def build_context_block(query: str) -> str:
    """Format retrieved chunks into a block to prepend to the LLM prompt."""
    hits = retrieve(query)
    if not hits:
        return ""
    lines = ["Relevant context from the knowledge base:"]
    for h in hits:
        lines.append(f"- ({h['source']}) {h['text']}")
    return "\n".join(lines)
