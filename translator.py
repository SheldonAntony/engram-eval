#!/usr/bin/env python3
"""Translate plain-language answers into concrete technical values.

Used by the slot-filling flow when a user answers a plain-language question
(e.g. "quick tests") and the plugin needs a technical value (e.g. "pytest").

Returns None when no API key is available — the TypeScript caller falls back
to asking a direct technical question instead.

CLI usage:
    translator.py '{"slot": "testing_framework", "answer": "quick simple tests",
                    "context": ["Python", "FastAPI"], "apiKey": "sk-..."}'

Returns:
    {"value": "<technical_value>"}  — or {"value": null} when no key supplied
"""

import json
import sys


def translate(
    project_context: list[str],
    user_answer: str,
    slot: str,
    api_key: str | None,
) -> str | None:
    """Return a technical value for *slot*, or None if no API key is available."""
    # No key → graceful degradation; caller will ask a direct technical question
    if not api_key:
        return None
    try:
        from anthropic import Anthropic  # lazy import

        client = Anthropic(api_key=api_key)
        context_str = "; ".join(project_context) if project_context else "unknown"

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=20,
            system=(
                "You are a technical translator for a coding assistant. "
                "Given project context and a user's plain-language answer, "
                "return only the specific technical value that best fits. "
                "One word or short phrase only. No explanation."
            ),
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Project context: {context_str}. "
                        f"User answered: {user_answer}. "
                        f"Slot needed: {slot}."
                    ),
                }
            ],
        )
        return response.content[0].text.strip()
    except Exception as e:
        print(f"TRANSLATE_ERROR: {e}", file=sys.stderr)
        return None


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Missing JSON argument"}), file=sys.stderr)
        sys.exit(1)

    data = json.loads(sys.argv[1])
    # api_key may be absent or null — translate() handles both gracefully
    api_key = data.get("apiKey")
    result = translate(
        data.get("context", []),
        data["answer"],
        data["slot"],
        api_key,
    )
    # result is str | None; null signals "no translation available" to caller
    print(json.dumps({"value": result}))
