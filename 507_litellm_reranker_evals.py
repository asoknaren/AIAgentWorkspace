"""Compare sparse (BM25), dense (embeddings), hybrid (RRF) and hybrid + LLM reranker retrieval
on past support tickets, using Recall@K, Precision@K, MRR, MAP, nDCG@K and AUROC."""
from dotenv import load_dotenv

from evals import evaluate
from retrieval import TicketRetriever, rank_of
from tickets_data import ADVERSARIAL_QUERIES, EVAL_QUERIES, PAST_TICKETS

load_dotenv()  # expects OPENAI_API_KEY in environment/.env

K = 3


def print_table(results: dict[str, dict[str, float]]) -> None:
    metrics = list(next(iter(results.values())))
    print(f"{'method':<18}" + "".join(f"{m:>14}" for m in metrics))
    for name, vals in results.items():
        print(f"{name:<18}" + "".join(f"{vals[m]:>14.3f}" for m in metrics))


def print_per_query(retriever: TicketRetriever) -> None:
    print("\nTop-1 per query (OK = relevant):")
    methods = retriever.methods()
    print(f"{'query':<58}" + "".join(f"{n:<18}" for n in methods))
    for q in EVAL_QUERIES:
        cells = []
        for fn in methods.values():
            top = rank_of(fn(q["query"]))[0]
            cells.append(f"{top} {'OK' if top in q['relevant'] else 'x'}")
        print(f"{q['query'][:56]:<58}" + "".join(f"{c:<18}" for c in cells))


def print_adversarial(retriever: TicketRetriever) -> None:
    print("\nAdversarial queries: rank of the best relevant ticket (1 = top; lower is better):")
    methods = retriever.methods()
    print(f"{'query':<46}" + "".join(f"{n:<18}" for n in methods))
    for q in ADVERSARIAL_QUERIES:
        cells = []
        for fn in methods.values():
            ranked = rank_of(fn(q["query"]))
            cells.append(str(min(ranked.index(t) for t in q["relevant"]) + 1))
        print(f"{q['query'][:44]:<46}" + "".join(f"{c:<18}" for c in cells))
        print(f"    hypothesis: {q['expect']}")


if __name__ == "__main__":
    retriever = TicketRetriever(PAST_TICKETS)
    results = {name: evaluate(fn, EVAL_QUERIES, K) for name, fn in retriever.methods().items()}
    print(f"Queries: {len(EVAL_QUERIES)}, tickets: {len(PAST_TICKETS)}, K={K}\n")
    print_table(results)
    print_per_query(retriever)
    print_adversarial(retriever)
    print("\nReading the results:")
    print("- Sparse nails exact identifiers (ERR_5021, SAML_403, HTTP 429) but misses paraphrases.")
    print("- Dense handles paraphrases ('erase everything' -> GDPR delete) but can blur exact codes.")
    print("- Hybrid (RRF) keeps both strengths; the reranker then fixes ordering within the top candidates.")
