"""Email conversation with memory: multiple LiteLLM calls share a growing message history."""
from dotenv import load_dotenv
from litellm import completion

load_dotenv()  # expects OPENAI_API_KEY in environment/.env

MODEL = "gpt-4o"

SYSTEM_PROMPT = (
    "You are an email assistant. The user shares emails and asks follow-up questions. "
    "Use the earlier conversation to answer; be concise."
)

EMAIL = """From: Priya Sharma <priya.sharma@example.com>
To: team@example.com
Subject: Q3 launch update

Hi team,

The Q3 launch is moving to October 28 because QA found a payment bug.
Please finish your regression tests by Friday and report blockers in Slack.

Thanks,
Priya
"""


class EmailConversation:
    def __init__(self, model: str = MODEL):
        self.model = model
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    def ask(self, user_text: str) -> str:
        self.messages.append({"role": "user", "content": user_text})
        response = completion(model=self.model, messages=self.messages, temperature=0)
        answer = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": answer})
        return answer


if __name__ == "__main__":
    chat = EmailConversation()
    turns = [
        f"Here is an email:\n\n{EMAIL}\nSummarize it in one sentence.",
        "Who sent it?",  # relies on memory of the email
        "What is the new launch date and why did it change?",
        "Draft a short reply from me confirming I'll finish my tests by Friday.",
        "What was the first question I asked you?",  # tests memory of the conversation itself
    ]
    for i, q in enumerate(turns, 1):
        print(f"\n--- Call {i} (history: {len(chat.messages)} messages) ---")
        print(f"USER: {q.splitlines()[0]}")
        print(f"LLM : {chat.ask(q)}")

    print("\n--- Stateless comparison (no memory) ---")
    stateless = completion(
        model=MODEL,
        messages=[{"role": "user", "content": "Who sent it?"}],
        temperature=0,
    )
    print(f"LLM : {stateless.choices[0].message.content}")
