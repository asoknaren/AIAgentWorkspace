"""Ranking metrics (Recall@K, Precision@K, MRR, MAP, nDCG@K, AUROC) and a query-set evaluator."""
import math

import numpy as np

from retrieval import rank_of


def recall_at_k(ranked, relevant, k):
    return len(set(ranked[:k]) & relevant) / len(relevant)


def precision_at_k(ranked, relevant, k):
    return len(set(ranked[:k]) & relevant) / k


def reciprocal_rank(ranked, relevant):
    for i, tid in enumerate(ranked, start=1):
        if tid in relevant:
            return 1 / i
    return 0.0


def average_precision(ranked, relevant):
    hits, total = 0, 0.0
    for i, tid in enumerate(ranked, start=1):
        if tid in relevant:
            hits += 1
            total += hits / i
    return total / len(relevant)


def ndcg_at_k(ranked, relevant, k):
    dcg = sum(1 / math.log2(i + 1) for i, tid in enumerate(ranked[:k], start=1) if tid in relevant)
    ideal = sum(1 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal


def auroc(scores, relevant):
    """P(random relevant ticket scores above random non-relevant one); ties count half."""
    pos = [s for t, s in scores.items() if t in relevant]
    neg = [s for t, s in scores.items() if t not in relevant]
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def evaluate(search, queries: list[dict], k: int) -> dict[str, float]:
    """Mean of each metric over queries; `search(query)` returns {ticket_id: score}."""
    rows = []
    for q in queries:
        scores, rel = search(q["query"]), q["relevant"]
        ranked = rank_of(scores)
        rows.append({
            f"Recall@{k}": recall_at_k(ranked, rel, k),
            f"Precision@{k}": precision_at_k(ranked, rel, k),
            "MRR": reciprocal_rank(ranked, rel),
            "MAP": average_precision(ranked, rel),
            f"nDCG@{k}": ndcg_at_k(ranked, rel, k),
            "AUROC": auroc(scores, rel),
        })
    return {m: float(np.mean([r[m] for r in rows])) for m in rows[0]}
