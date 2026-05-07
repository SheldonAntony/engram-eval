## Preflight Memory — How To Use

At the start of every session:
1. Call `get_project_id` with your current working directory
2. Call `get_context` with the user's first prompt, the session ID,
   and the project ID
3. Use the returned memories, similar_tasks, and slots as context
4. For each slot in `missing_slots`, ask the user one plain-language
   question and call `store_slot` with their answer before proceeding

During the session:
- When the user states something important about the codebase,
  architecture, or workflow: call `store_memory`
- When the user corrects something you assumed: call `store_memory`
  with the correction, then call `store_slot` if it is a config value
- When the user states a personal preference (response style,
  language preferences, conventions): call `store_memory` with
  fact_type="preference" — this saves to global memory across projects

Do not ask for slots already returned in the `slots` field of `get_context`.
Do not call `get_context` more than once per session unless the conversation
window is compacted (in which case call it again on the next message).
