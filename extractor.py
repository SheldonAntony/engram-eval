#!/usr/bin/env python3
"""LLM council: extract memorable facts from session summaries.

The "council" reviews what happened in a session and decides which facts
are worth storing in long-term memory — 1 to 3 concise, reusable statements.

CLI usage:
    extractor.py '{"text": "...", "apiKey": "sk-..."}'

Returns:
    {"facts": ["fact1", "fact2"]}    — extracted facts (array, may be empty)
    {"facts": []}                    — if text is too short or trivial
"""

import json
import sys


# ─── Keyword-based fallback ───────────────────────────────────────────────────

# Minimum token count before we bother trying to extract anything
_MIN_WORDS = 15

# Markers that hint at something memorable happened
_SIGNAL_PATTERNS = [
    "decided", "chose", "fixed", "solved", "discovered", "realized",
    "switched", "migrated", "refactored", "added", "removed", "implemented",
    "use ", "using ", "framework", "database", "api", "endpoint",
    "auth", "error", "bug", "pattern", "architecture",
]


def keyword_extract(text: str) -> list[str]:
    """Very cheap extraction: return the text itself if it looks signal-rich."""
    words = text.split()
    if len(words) < _MIN_WORDS:
        return []
    lower = text.lower()
    hits = sum(1 for p in _SIGNAL_PATTERNS if p in lower)
    if hits < 2:
        return []
    # Trim to a single concise sentence if very long
    sentences = [s.strip() for s in text.replace("\n", ". ").split(".") if s.strip()]
    return [sentences[0][:200]] if sentences else []


# ─── LLM council ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a memory curator for a software developer's AI coding assistant.
Given a summary of what happened during a coding session, extract 1-3 concise
facts that are worth remembering for future sessions.

Rules:
- Each fact must be a single sentence, max 120 characters.
- Only include facts that would be useful context in a future unrelated session.
- Skip implementation details that won't matter next time.
- Skip questions the user asked. Focus on decisions, findings, and outcomes.
- If nothing is worth remembering, return an empty list.

Respond with a JSON object: {"facts": ["...", "..."]}
Do not add any text outside the JSON."""


def llm_extract(text: str, api_key: str | None) -> list[str]:
    """Ask Claude Haiku to council what facts deserve long-term storage."""
    if not api_key or len(text.split()) < _MIN_WORDS:
        return keyword_extract(text)
    try:
        from anthropic import Anthropic  # lazy import

        client = Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=256,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": text[:2000]}],
        )
        raw = response.content[0].text.strip()
        parsed = json.loads(raw)
        facts = parsed.get("facts", [])
        # Sanitize: only strings, max 3, max 160 chars each
        return [str(f)[:160] for f in facts if isinstance(f, str)][:3]
    except Exception:
        # Council unavailable — fall back to keyword extraction
        return keyword_extract(text)


# ─── CLI entry point ──────────────────────────────────────────────────────────

def main() -> None:
    if len(sys.argv) < 2:
        print(json.dumps({"error": "usage: extractor.py '{\"text\": \"...\", \"apiKey\": \"...\"}'"}))
        sys.exit(1)

    try:
        payload = json.loads(sys.argv[1])
    except json.JSONDecodeError as exc:
        print(json.dumps({"error": f"JSON parse error: {exc}"}))
        sys.exit(1)

    text    = payload.get("text", "").strip()
    api_key = payload.get("apiKey") or None

    facts = llm_extract(text, api_key)
    print(json.dumps({"facts": facts}))


if __name__ == "__main__":
    main()
