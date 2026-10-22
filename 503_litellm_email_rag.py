"""Email RAG: embed emails with LiteLLM (text-embedding-3-small), keep vectors in memory,
retrieve by cosine similarity, and add the related emails to the user prompt."""
import math

from dotenv import load_dotenv
from litellm import completion, embedding

load_dotenv()  # expects OPENAI_API_KEY in environment/.env

CHAT_MODEL = "gpt-4o"
EMBED_MODEL = "text-embedding-3-small"

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


def email_text(e: dict) -> str:
    return f"From: {e['from']}\nSubject: {e['subject']}\n\n{e['body']}"


def embed(texts: list[str]) -> list[list[float]]:
    response = embedding(model=EMBED_MODEL, input=texts)
    return [item["embedding"] for item in response.data]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


class EmailIndex:
    """In-memory vector store."""

    def __init__(self, emails: list[dict]):
        self.emails = emails
        self.vectors = embed([email_text(e) for e in emails])

    def search(self, query: str, k: int = 3) -> list[tuple[float, dict]]:
        qv = embed([query])[0]
        scored = [(cosine_similarity(qv, v), e) for v, e in zip(self.vectors, self.emails)]
        return sorted(scored, key=lambda s: s[0], reverse=True)[:k]


def answer(index: EmailIndex, question: str, k: int = 3) -> str:
    hits = index.search(question, k)
    related = "\n\n".join(
        f"[Email {e['id']}] (similarity {score:.3f})\n{email_text(e)}" for score, e in hits
    )
    print("Retrieved:", [(e["id"], e["subject"], round(s, 3)) for s, e in hits])
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
    index = EmailIndex(EMAILS)
    for q in [
        "Why was the launch delayed and what caused the bug?",
        "What do I need to test before the release?",
        "When is the holiday party?",
    ]:
        print(f"\nQ: {q}")
        print(f"A: {answer(index, q)}")
