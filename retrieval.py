"""Sparse (BM25), dense (embeddings), hybrid (RRF) and LLM-reranked retrieval over support tickets.

Every method returns {ticket_id: score} for the whole corpus so metrics like AUROC can use full rankings.
"""
import json
import re

import numpy as np
from litellm import completion, embedding
from rank_bm25 import BM25Okapi

EMBED_MODEL = "text-embedding-3-small"
RERANK_MODEL = "gpt-4o-mini"
RRF_K = 60


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", text.lower())


def rank_of(scores: dict[str, float]) -> list[str]:
    return sorted(scores, key=scores.get, reverse=True)


class TicketRetriever:
    def __init__(self, tickets: list[dict]):
        self.tickets = tickets
        self.ids = [t["id"] for t in tickets]
        self.text = {t["id"]: t["text"] for t in tickets}
        self.bm25 = BM25Okapi([tokenize(t["text"]) for t in tickets])
        self.doc_vecs = self._embed([t["text"] for t in tickets])

    @staticmethod
    def _embed(texts: list[str]) -> np.ndarray:
        data = embedding(model=EMBED_MODEL, input=texts).data
        vecs = np.array([d["embedding"] for d in data])
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

    def sparse(self, query: str) -> dict[str, float]:
        return dict(zip(self.ids, self.bm25.get_scores(tokenize(query)).tolist()))

    def dense(self, query: str) -> dict[str, float]:
        sims = self.doc_vecs @ self._embed([query])[0]  # cosine: vectors are unit length
        return dict(zip(self.ids, sims.tolist()))

    def hybrid(self, query: str) -> dict[str, float]:
        """Reciprocal Rank Fusion: scale-free, so BM25 and cosine scores need no normalization."""
        fused = dict.fromkeys(self.ids, 0.0)
        for scores in (self.sparse(query), self.dense(query)):
            for rank, tid in enumerate(rank_of(scores), start=1):
                fused[tid] += 1.0 / (RRF_K + rank)
        return fused

    def rerank(self, query: str, top_n: int = 6) -> dict[str, float]:
        """Hybrid candidates re-scored by an LLM cross-encoder-style judge; the rest keep hybrid order below them."""
        fused = self.hybrid(query)
        candidates = rank_of(fused)[:top_n]
        listing = "\n".join(f'{tid}: {self.text[tid]}' for tid in candidates)
        prompt = (
            "Rate how relevant each past support ticket is to the new query on a 0-10 scale "
            "(10 = same underlying issue, 0 = unrelated).\n"
            f"Query: {query}\n\nTickets:\n{listing}\n\n"
            'Reply with JSON only: {"scores": {"<ticket_id>": <number>, ...}}'
        )
        raw = completion(model=RERANK_MODEL, temperature=0,
                         response_format={"type": "json_object"},
                         messages=[{"role": "user", "content": prompt}]).choices[0].message.content
        llm_scores = json.loads(raw).get("scores", {})
        out = dict(fused)  # non-candidates stay in [0, ~0.033], below every candidate
        for tid in candidates:
            out[tid] = 1000 + float(llm_scores.get(tid, 0)) + fused[tid]  # fused score breaks ties
        return out

    def methods(self) -> dict:
        return {"sparse (BM25)": self.sparse, "dense (embed)": self.dense,
                "hybrid (RRF)": self.hybrid, "hybrid + rerank": self.rerank}
