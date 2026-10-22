"""Email tool call: the LLM extracts phone numbers from an email and calls a local
normalize_phone tool (E.164 output), then uses the tool results in its final answer."""
import json
import re

from dotenv import load_dotenv
from litellm import completion

load_dotenv()  # expects OPENAI_API_KEY in environment/.env

MODEL = "gpt-4o"

EMAIL = """From: Priya Sharma <priya.sharma@example.com>
To: team@example.com
Subject: Contacts for the launch

Hi team,

If the payment issue comes up, call Marcus at (415) 555-0132 or my desk at 415.555.0187 x22.
For the London vendor, use +44 20 7946 0958. Hannah's cell is 1-212-555-0199.

Thanks,
Priya
"""


def normalize_phone(phone: str, default_country_code: str = "1") -> dict:
    """Normalize to E.164; keeps an extension separately."""
    ext_match = re.search(r"(?:x|ext\.?)\s*(\d+)\s*$", phone, flags=re.I)
    extension = ext_match.group(1) if ext_match else None
    base = phone[: ext_match.start()] if ext_match else phone
    has_plus = base.strip().startswith("+")
    digits = re.sub(r"\D", "", base)
    if not has_plus:
        if len(digits) == 10:
            digits = default_country_code + digits
        elif not (len(digits) == 11 and digits.startswith(default_country_code)):
            return {"input": phone, "valid": False, "error": "unrecognized format"}
    if not 8 <= len(digits) <= 15:
        return {"input": phone, "valid": False, "error": "invalid length"}
    return {"input": phone, "valid": True, "e164": f"+{digits}", "extension": extension}


TOOLS = [{
    "type": "function",
    "function": {
        "name": "normalize_phone",
        "description": "Normalize a phone number to E.164 format (e.g. +14155550132).",
        "parameters": {
            "type": "object",
            "properties": {
                "phone": {"type": "string", "description": "Phone number exactly as written in the email"},
                "default_country_code": {"type": "string", "description": "Country code if none given", "default": "1"},
            },
            "required": ["phone"],
        },
    },
}]
FUNCTIONS = {"normalize_phone": normalize_phone}


def run(email_text: str) -> str:
    messages = [
        {"role": "system", "content": "You extract contacts from emails. Always use the normalize_phone "
                                      "tool for every phone number; never normalize numbers yourself."},
        {"role": "user", "content": f"List each contact name with their normalized phone number:\n\n{email_text}"},
    ]
    
    while True:

        print("Model Call START" + "-" * 100)
        # Get the next message from the model, which may include tool calls
        message = completion(model=MODEL, messages=messages, tools=TOOLS, temperature=0).choices[0].message

        print(f"MODEL: {message.content}")
        messages.append(message.model_dump(exclude_none=True))

        if not message.tool_calls:
            print("-" * 100)
            print("MODEL did not make any tool calls.")
            print("-" * 100)
            return message.content

        # Process each tool call made by the model
        for call in message.tool_calls:
            args = json.loads(call.function.arguments)
            result = FUNCTIONS[call.function.name](**args)
            print(f"TOOL {call.function.name}({args}) -> {result}")
            messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)})


if __name__ == "__main__":
    print(run(EMAIL))
