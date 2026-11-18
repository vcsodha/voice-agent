import os

import rag


class FakeCollection:
    """Tiny stand-in for a chroma collection."""

    def __init__(self, docs=None, stored=None):
        self.docs = docs or []      # list of (text, source, distance) for query()
        self.stored = dict(stored or {})   # id -> text, what's "in the database"
        self.upserts = []
        self.last_n = None

    def count(self):
        return len(self.docs)

    def get(self):
        return {"ids": list(self.stored)}

    def delete(self, ids):
        for i in ids:
            self.stored.pop(i, None)

    def query(self, query_texts, n_results):
        self.last_n = n_results
        picked = self.docs[:n_results]
        return {
            "documents": [[d[0] for d in picked]],
            "metadatas": [[{"source": d[1]} for d in picked]],
            "distances": [[d[2] for d in picked]],
        }

    def upsert(self, documents, ids, metadatas):
        self.upserts.append((documents, ids, metadatas))
        for i, d in zip(ids, documents):
            self.stored[i] = d


# --- chunking ---

def test_short_text_is_a_single_chunk():
    assert rag._chunk_text("hello there") == ["hello there"]


def test_long_text_is_split_with_overlap():
    text = "".join(chr(97 + i % 26) for i in range(2000))
    chunks = rag._chunk_text(text)
    assert len(chunks) == 3
    assert all(len(c) <= 800 for c in chunks)
    assert chunks[0][-100:] == chunks[1][:100]
    assert chunks[-1].endswith(text[-50:])      # nothing dropped at the end


def test_blank_text_gives_no_chunks():
    assert rag._chunk_text("") == []
    assert rag._chunk_text("   \n  ") == []


# --- retrieval ---

def test_retrieve_on_empty_knowledge_base(monkeypatch):
    monkeypatch.setattr(rag, "_get_collection", lambda: FakeCollection())
    assert rag.retrieve("anything") == []


def test_retrieve_returns_text_source_and_distance(monkeypatch):
    fake = FakeCollection([("Piper is the TTS.", "stack.txt", 0.2), ("Whisper does STT.", "stack.txt", 0.4)])
    monkeypatch.setattr(rag, "_get_collection", lambda: fake)
    hits = rag.retrieve("what does tts", top_k=2)
    assert hits == [
        {"text": "Piper is the TTS.", "source": "stack.txt", "distance": 0.2},
        {"text": "Whisper does STT.", "source": "stack.txt", "distance": 0.4},
    ]


def test_retrieve_never_asks_for_more_than_exists(monkeypatch):
    fake = FakeCollection([("only one", "a.txt", 0.1)])
    monkeypatch.setattr(rag, "_get_collection", lambda: fake)
    rag.retrieve("q", top_k=5)
    assert fake.last_n == 1


def test_context_block_is_empty_when_nothing_found(monkeypatch):
    monkeypatch.setattr(rag, "_get_collection", lambda: FakeCollection())
    assert rag.build_context_block("anything") == ""


def test_context_block_lists_sources(monkeypatch):
    fake = FakeCollection([("Barge-in cancels generation.", "about.txt", 0.1)])
    monkeypatch.setattr(rag, "_get_collection", lambda: fake)
    block = rag.build_context_block("what is barge in")
    assert block.splitlines()[0] == "Relevant context from the knowledge base:"
    assert "- (about.txt) Barge-in cancels generation." in block


# --- ingestion ---

def test_ingest_reads_txt_and_md_only_with_stable_ids(monkeypatch, tmp_path):
    (tmp_path / "a.txt").write_text("hello from a")
    (tmp_path / "b.md").write_text("hello from b")
    (tmp_path / "c.pdf").write_text("should be ignored")
    fake = FakeCollection()
    monkeypatch.setattr(rag, "_get_collection", lambda: fake)

    total = rag.ingest_knowledge_base(str(tmp_path))

    assert total == 2
    all_ids = sorted(i for _, ids, _ in fake.upserts for i in ids)
    assert all_ids == ["a.txt::0", "b.md::0"]      # same ids every run -> idempotent


def test_ingest_clears_chunks_from_files_that_no_longer_exist(monkeypatch, tmp_path):
    (tmp_path / "new.txt").write_text("fresh content")
    fake = FakeCollection(stored={"deleted_file.txt::0": "stale content"})
    monkeypatch.setattr(rag, "_get_collection", lambda: fake)

    rag.ingest_knowledge_base(str(tmp_path))

    assert list(fake.stored) == ["new.txt::0"]      # the stale chunk is gone


def test_same_file_name_in_two_folders_does_not_collide(monkeypatch, tmp_path):
    for folder in ("a", "b"):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / "notes.txt").write_text(f"notes from {folder}")
    fake = FakeCollection()
    monkeypatch.setattr(rag, "_get_collection", lambda: fake)

    rag.ingest_knowledge_base(str(tmp_path))

    assert sorted(fake.stored) == sorted([
        os.path.join("a", "notes.txt") + "::0",
        os.path.join("b", "notes.txt") + "::0",
    ])
