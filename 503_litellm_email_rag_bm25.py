"""Email RAG with BM25 keyword retrieval (rank_bm25, no embeddings) + LiteLLM for the answer."""
import re

from dotenv import load_dotenv
from litellm import completion
from rank_bm25 import BM25Okapi

load_dotenv()  # expects OPENAI_API_KEY in environment/.env

CHAT_MODEL = "gpt-4o"

EMAILS = [
    {"id": 1, "from": "Priya Sharma", "subject": "Q3 launch update",
     "body": "The Q3 launch is moving to October 28 because QA found a payment bug. "
             "Finish regression tests by Friday and report blockers in Slack."},
    {"id": 2, "from": "Marcus Lee", "subject": "Payment bug root cause",
     "body": "The payment bug was caused by a rounding error in the currency conversion module. "
             "A fix is merged to the release branch and awaiting QA verification."},
    {"id": 3, "from": "Hannah Cole", "subject": "Office holiday party",
     "body": "The holiday party is on December 12 at 6pm in the main cafeteria. "
             "Please RSVP and list any dietary restrictions."},
    {"id": 4, "from": "Dev Patel", "subject": "Regression test checklist",
     "body": "Attached is the regression checklist for the Q3 release: checkout, refunds, "
             "invoices, and currency conversion. Log failures in the QA tracker."},
    {"id": 5, "from": "Sofia Rossi", "subject": "Parking permit renewal",
     "body": "Parking permits expire at the end of the month. Renew through the facilities portal."},
]

STOPWORDS = {"the", "a", "an", "is", "are", "was", "were", "of", "to", "in", "on", "and", "or",
             "what", "when", "why", "who", "do", "i", "for", "by", "it", "that", "this", "me"}


def email_text(e: dict) -> str:
    return f"From: {e['from']}\nSubject: {e['subject']}\n\n{e['body']}"


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


class BM25EmailIndex:
    def __init__(self, emails: list[dict]):
        self.emails = emails
        self.bm25 = BM25Okapi([tokenize(email_text(e)) for e in emails])

    def search(self, query: str, k: int = 3) -> list[tuple[float, dict]]:
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(zip(scores, self.emails), key=lambda s: s[0], reverse=True)
        return [(float(s), e) for s, e in ranked[:k] if s > 0]


def answer(index: BM25EmailIndex, question: str, k: int = 3) -> str:
    hits = index.search(question, k)
    print("Retrieved:", [(e["id"], e["subject"], round(s, 3)) for s, e in hits])
    print("-" * 100)
    
    if not hits:
        return "No related emails found."
    related = "\n\n".join(f"[Email {e['id']}] (bm25 {s:.3f})\n{email_text(e)}" for s, e in hits)
    user_prompt = (
        f"Related emails:\n{related}\n\n"
        f"Question: {question}\n"
        "Answer using only the related emails above and cite email ids."
    )
    response = completion(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": "You are an assistant that answers questions about a mailbox."},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    index = BM25EmailIndex(EMAILS)
    for q in [
        "Why was the launch delayed and what caused the payment bug?",
        "Which regression tests do I need to run?",
        "When is the holiday party?",
    ]:
        print(f"\nQ: {q}")
        print(f"A: {answer(index, q)}")
