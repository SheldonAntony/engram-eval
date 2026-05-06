## Preflight — Project Context

When the enriched prompt contains a `## Project context` section, use those slot values throughout your response.

If that section is **missing slots** (framework, database, testing_framework, language, etc.) that are relevant to the task:
1. Ask the user a single plain-language question for each missing slot
2. Once they answer, call `preflight_store_slot` with the session ID, slot name, and value before proceeding

When the user tells you something important about the codebase or workflow (architecture decisions, conventions, recurring patterns), call `preflight_store_memory` to save it.

Do not ask for slots that are already in `## Project context` or that are clearly irrelevant to the current task.
