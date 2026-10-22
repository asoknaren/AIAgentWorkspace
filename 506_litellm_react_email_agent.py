"""ReAct-style support email summarizer: text-based Thought/Action/Observation loop with LiteLLM,
BM25 ticket retrieval, and a normalize_phone formatting tool."""
import json
import re

from dotenv import load_dotenv
from litellm import completion
from rank_bm25 import BM25Okapi

load_dotenv()  # expects OPENAI_API_KEY in environment/.env

MODEL = "gpt-4o"
MAX_STEPS = 6

SYSTEM_PROMPT = """### Role
You are an agent that helps summarize support emails.

### Objective
Read a new email, use retrieval and tools when helpful, and produce a final summary for the support team.

### Tools
- retrieve_tickets: search past support tickets. Args: {"query": "<text>", "sender": "<sender_email>"}
- normalize_phone: normalize a phone number to E.164. Args: {"number": "<phone as written>"}

### Instructions
- You may retrieve past tickets if needed.
- You may call `normalize_phone` when a phone number is present.
- Work step by step. At each step, decide on one action (retrieve, call a tool, or stop with a final answer).
- After you take an Action, you will receive an Observation with the result. Use it to decide your next step.
- End with a final answer in the specified format.
- If the user's request is ambiguous, infer the most likely intent and continue.

### Reasoning (ReAct pattern)
Follow the Thought -> Action -> Observation cycle:
- Thought: a brief statement of what you intend to do.
- Action: the tool name to call.
- Args: JSON arguments on a single line.
- Observation: will be provided after each action. Never write it yourself.
Emit exactly one Action per step, then stop and wait for the Observation.

### Output format (for Final)
Final answer:
- Sender: <name>
- Requests:
  - <bullet 1>
  - <bullet 2>
  - <bullet 3>
- Phone: <normalized number if present>

### Step format example
Thought: I should check prior tickets for context.
Action: retrieve_tickets
Args: {"query": "can't log in; password reset", "sender": "<sender_email>"}

Observation: [{"id":"4521","text":"Please reset my password. I cannot log in."}]

Thought: A phone number is present; I will normalize it.
Action: normalize_phone
Args: {"number": "(415) 555-9021 ext. 7"}

Observation: "+14155559021"

Thought: I have enough information to write the summary.
Final answer:
- Sender: Alice Johnson
- Requests:
  - Reset account access
  - Verify email on file
  - Call back
- Phone: +14155559021
"""

PAST_TICKETS = [
    {"id": "4521", "sender": "alice.johnson@example.com",
     "text": "Please reset my password. I cannot log in to my account."},
    {"id": "4533", "sender": "alice.johnson@example.com",
     "text": "Verification email never arrived; please verify the email address on file."},
    {"id": "4602", "sender": "bob.lee@example.com",
     "text": "Invoice for March shows a duplicate charge. Requesting a refund."},
    {"id": "4650", "sender": "carol.diaz@example.com",
     "text": "Our API keys stopped working after the migration. Need new keys issued."},
]


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


BM25 = BM25Okapi([tokenize(t["text"]) for t in PAST_TICKETS])


def retrieve_tickets(query: str, sender: str | None = None, k: int = 3) -> list[dict]:
    scores = BM25.get_scores(tokenize(query))
    ranked = sorted(zip(scores, PAST_TICKETS), key=lambda s: s[0], reverse=True)
    hits = [t for s, t in ranked if s > 0 or (sender and t["sender"] == sender)]
    if sender:  # prefer the same sender's history
        hits.sort(key=lambda t: t["sender"] != sender)
    return [{"id": t["id"], "text": t["text"]} for t in hits[:k]]


def normalize_phone(number: str, default_country_code: str = "1") -> str:
    ext = re.search(r"(?:x|ext\.?)\s*\d+\s*$", number, flags=re.I)
    base = number[: ext.start()] if ext else number
    digits = re.sub(r"\D", "", base)
    if not base.strip().startswith("+"):
        if len(digits) == 10:
            digits = default_country_code + digits
        elif not (len(digits) == 11 and digits.startswith(default_country_code)):
            return "ERROR: unrecognized phone format"
    return f"+{digits}"


TOOLS = {"retrieve_tickets": retrieve_tickets, "normalize_phone": normalize_phone}
ACTION_RE = re.compile(r"Action:\s*(\w+)\s*\nArgs:\s*(\{.*\})", re.S)


def run_agent(email_text: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Summarize this new support email. Check the sender's past tickets for context first.\n\n{email_text}"},
    ]
    for step in range(1, MAX_STEPS + 1):
        # stop at "Observation:" so the model cannot invent tool results
        out = completion(model=MODEL, messages=messages, temperature=0,
                         stop=["Observation:"]).choices[0].message.content.strip()
        print(f"\n--- Step {step} ---\n{out}")
        messages.append({"role": "assistant", "content": out})

        if "Final answer:" in out:
            return out[out.index("Final answer:"):]

        match = ACTION_RE.search(out)
        if not match:
            observation = "ERROR: respond with an Action and Args, or a Final answer."
        else:
            name, raw_args = match.group(1), match.group(2).strip()
            try:
                observation = json.dumps(TOOLS[name](**json.loads(raw_args)))
            except Exception as exc:  # feed errors back so the agent can recover
                observation = f"ERROR: {exc}"
        print(f"Observation: {observation}")
        messages.append({"role": "user", "content": f"Observation: {observation}"})
    return "Stopped: max steps reached without a final answer."


if __name__ == "__main__":
    EMAIL = """From: Alice Johnson <alice.johnson@example.com>
Subject: Still can't get in

Hi support,

I still can't log in even after the reset, and I never got the verification email.
Could someone call me back? My number is (415) 555-9021 ext. 7.

Thanks,
Alice
"""
    result = run_agent(EMAIL)
    print("\n=== FINAL ===")
    print(result)
