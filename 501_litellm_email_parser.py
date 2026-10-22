"""Email parser client: extracts sender name and summarizes an email using LiteLLM + gpt-4o."""
import json

from dotenv import load_dotenv
from litellm import completion

load_dotenv()  # expects OPENAI_API_KEY in environment/.env

MODEL = "gpt-4o"

SYSTEM_PROMPT = (
    "You are an email parser. Given a raw email, return JSON with exactly these keys: "
    '"sender_name" (the full name of the person who sent it; null if unknown) and '
    '"summary" (a concise 1-3 sentence summary of the email).'
)


def parse_email(email_text: str, model: str = MODEL) -> dict:
    response = completion(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": email_text},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return json.loads(response.choices[0].message.content)


if __name__ == "__main__":
    sample = """From: Priya Sharma <priya.sharma@example.com>
To: team@example.com
Subject: Q3 launch update

Hi team,

The Q3 launch is moving to October 28 because QA found a payment bug.
Please finish your regression tests by Friday and report blockers in Slack.

Thanks,
Priya
"""
    print(json.dumps(parse_email(sample), indent=2))
