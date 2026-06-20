from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from riskops_ai.config import POLICY_DIR, INDEX_DIR, ensure_dirs

INDEX_FILE = INDEX_DIR / "policy_tfidf_index.pkl"

@dataclass
class RetrievedChunk:
    source: str
    text: str
    score: float


def _chunk_text(text: str, max_words: int = 120, overlap: int = 25) -> list[str]:
    words = re.findall(r"\S+", text)
    chunks = []
    step = max_words - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + max_words])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def load_policy_chunks(policy_dir: Path = POLICY_DIR) -> list[dict[str, str]]:
    chunks: list[dict[str, str]] = []
    for path in sorted(policy_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        for idx, chunk in enumerate(_chunk_text(text)):
            chunks.append({"source": f"{path.name}#chunk-{idx+1}", "text": chunk})
    return chunks


def build_index() -> int:
    ensure_dirs()
    chunks = load_policy_chunks()
    if not chunks:
        raise RuntimeError(f"No policy docs found in {POLICY_DIR}")
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=5000)
    matrix = vectorizer.fit_transform([c["text"] for c in chunks])
    with INDEX_FILE.open("wb") as f:
        pickle.dump({"chunks": chunks, "vectorizer": vectorizer, "matrix": matrix}, f)
    return len(chunks)


class PolicyRetriever:
    def __init__(self, index_file: Path = INDEX_FILE):
        if not index_file.exists():
            build_index()
        with index_file.open("rb") as f:
            payload = pickle.load(f)
        self.chunks = payload["chunks"]
        self.vectorizer = payload["vectorizer"]
        self.matrix = payload["matrix"]

    def retrieve(self, query: str, k: int = 3) -> list[RetrievedChunk]:
        q = self.vectorizer.transform([query])
        sims = cosine_similarity(q, self.matrix)[0]
        ranked = sims.argsort()[::-1][:k]
        return [
            RetrievedChunk(
                source=self.chunks[i]["source"],
                text=self.chunks[i]["text"],
                score=round(float(sims[i]), 4),
            )
            for i in ranked
        ]


if __name__ == "__main__":
    n = build_index()
    print(f"Built policy index with {n} chunks at {INDEX_FILE}")
