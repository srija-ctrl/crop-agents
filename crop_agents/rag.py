"""Step 4: small text-only knowledge base (RAG).

Backend 1 (preferred): ChromaDB + Ollama embeddings (nomic-embed-text).
Backend 2 (automatic fallback): TF-IDF, so the project still runs if Chroma / the embedding
model is missing.  The CSV rows are NEVER used as documents - only the .txt corpus.

You can drop your existing RAG corpus into data/knowledge/*.txt.  Files that use `## heading`
sections (one per crop) get crop-aware retrieval; plain paragraphs are also supported.
"""
from __future__ import annotations
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import (CHROMA_DIR, KNOWLEDGE_DIR, MIN_RAG_SCORE_CHROMA, MIN_RAG_SCORE_TFIDF)
from .llm import LLMUnavailable, OllamaLLM


@dataclass
class Chunk:
    uid: str      # unique id, e.g. agri_knowledge.txt#rice
    source: str
    crop: str     # lower-case heading or "general"
    text: str


def load_chunks(directory: Path = KNOWLEDGE_DIR) -> List[Chunk]:
    chunks: List[Chunk] = []
    for f in sorted(Path(directory).glob("*.txt")):
        text = f.read_text(encoding="utf-8")
        parts = re.split(r"^##\s+(.+)$", text, flags=re.M)
        if len(parts) > 1:  # sectioned file: parts = [preamble, head1, body1, head2, body2, ...]
            for i in range(1, len(parts), 2):
                head = parts[i].strip().lower()
                body = " ".join(parts[i + 1].split())
                if body:
                    n = sum(1 for c in chunks if c.crop == head and c.uid.startswith(f"{f.name}#{head}"))
                    uid = f"{f.name}#{head}" + (f"-{n + 1}" if n else "")
                    chunks.append(Chunk(uid, uid, head, body))
        else:  # plain paragraphs
            for n, para in enumerate(re.split(r"\n\s*\n", text), start=1):
                body = " ".join(para.split())
                if body:
                    chunks.append(Chunk(f"{f.name}#p{n}", f"{f.name}#p{n}", "general", body))
    return chunks


class KnowledgeBase:
    def __init__(self, llm: OllamaLLM, directory: Path = KNOWLEDGE_DIR, force_tfidf: bool = False):
        self.llm = llm
        self.directory = Path(directory)
        self.force_tfidf = force_tfidf
        self.chunks: List[Chunk] = []
        self.backend = "none"
        self._collection = None
        self._tfidf = None
        self._matrix = None

    # ---- build ------------------------------------------------------------
    def build(self, rebuild: bool = False) -> str:
        self.chunks = load_chunks(self.directory)
        if not self.chunks:
            self.backend = "none"
            return self.backend
        if not self.force_tfidf and self._try_chroma(rebuild):
            self.backend = "chroma+ollama-embeddings"
        else:
            self._build_tfidf()
            self.backend = "tfidf-fallback"
        return self.backend

    def _try_chroma(self, rebuild: bool) -> bool:
        try:
            import chromadb
            if not self.llm.embed_available():
                return False
            client = chromadb.PersistentClient(path=str(CHROMA_DIR))
            digest = hashlib.sha256(
                ("|".join(c.uid + c.text for c in self.chunks) + self.llm.embed_model).encode()
            ).hexdigest()[:16]
            name = "agri_knowledge"
            reuse = False
            if not rebuild:
                try:
                    col = client.get_collection(name)
                    reuse = (col.metadata or {}).get("corpus_hash") == digest and col.count() == len(self.chunks)
                except Exception:
                    reuse = False
            if reuse:
                self._collection = col
                return True
            try:
                client.delete_collection(name)
            except Exception:
                pass
            col = client.create_collection(name, metadata={"hnsw:space": "cosine", "corpus_hash": digest})
            embeddings = self.llm.embed([c.text for c in self.chunks])
            col.add(
                ids=[c.uid for c in self.chunks],
                documents=[c.text for c in self.chunks],
                embeddings=embeddings,
                metadatas=[{"crop": c.crop, "source": c.source} for c in self.chunks],
            )
            self._collection = col
            return True
        except (ImportError, LLMUnavailable):
            return False
        except Exception:
            return False

    def _build_tfidf(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        self._matrix = self._tfidf.fit_transform([f"{c.crop} {c.text}" for c in self.chunks])

    # ---- search -----------------------------------------------------------
    def search(self, query: str, k: int = 3, crops: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        if not self.chunks:
            return []
        crops = [c.lower() for c in crops] if crops else None
        if self.backend.startswith("chroma"):
            hits = self._search_chroma(query, k, crops)
        else:
            hits = self._search_tfidf(query, k, crops)
        return [{"id": f"S{i}", **h} for i, h in enumerate(hits, start=1)]

    def _search_chroma(self, query: str, k: int, crops: Optional[List[str]]) -> List[Dict[str, Any]]:
        emb = self.llm.embed([query])
        kwargs: Dict[str, Any] = {"query_embeddings": emb, "n_results": k}
        if crops:
            kwargs["where"] = {"crop": {"$in": crops}}
        res = self._collection.query(**kwargs)
        hits = []
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            score = 1.0 - float(dist)
            # similarity floor only for open-ended searches; a crop-filtered search is already restricted
            if crops or score >= MIN_RAG_SCORE_CHROMA:
                hits.append({"source": meta["source"], "crop": meta["crop"], "text": doc, "score": round(score, 3)})
        return hits

    def _search_tfidf(self, query: str, k: int, crops: Optional[List[str]]) -> List[Dict[str, Any]]:
        from sklearn.metrics.pairwise import cosine_similarity
        sims = cosine_similarity(self._tfidf.transform([query]), self._matrix)[0]
        order = sims.argsort()[::-1]
        hits = []
        for i in order:
            c = self.chunks[i]
            if crops and c.crop not in crops:
                continue
            if sims[i] < MIN_RAG_SCORE_TFIDF:
                continue
            hits.append({"source": c.source, "crop": c.crop, "text": c.text, "score": round(float(sims[i]), 3)})
            if len(hits) >= k:
                break
        return hits
