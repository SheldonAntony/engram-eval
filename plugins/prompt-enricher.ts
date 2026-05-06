/**
 * prompt-enricher.ts — Preflight: context enrichment plugin for opencode
 *
 * Intercepts every chat message and enriches it with:
 *   - Semantically similar past tasks (tasks.py)
 *   - Relevant facts from memory (memory.py)
 *   - Pre-filled slot values with interactive confirmation (Feature 6)
 *   - Intent-to-technical slot translation (Feature 8)
 *   - LLM-based task classification (Feature 5)
 *
 * Features implemented:
 *   Feature 1 — async Python bridge, session-scoped slot fills, shared utils.py
 *   Feature 2 — project-scoped memory via project_id
 *   Feature 3 — weighted retrieval (similarity + recency + frequency)
 *   Feature 4 — configurable confidence threshold via preflight.config.json
 *   Feature 5 — hybrid keyword + LLM classifier (classifier.py)
 *   Feature 6 — interactive slot-fill confirmation
 *   Feature 7 — snapshot recency decay in tasks.py
 *   Feature 8 — plain-language → technical slot translation (translator.py)
 */

import type { Plugin } from "@opencode-ai/plugin"
import { tool }       from "@opencode-ai/plugin"
import { execSync }   from "child_process"
import { spawn }      from "child_process"
import * as crypto    from "crypto"
import * as fs        from "fs"
import * as os        from "os"
import * as path      from "path"

// ─── Paths ────────────────────────────────────────────────────────────────────

const CONFIG_DIR = path.join(os.homedir(), ".config", "opencode")

/**
 * Path to the Python interpreter in the project venv.
 * Override with PREFLIGHT_PYTHON env var for non-standard setups.
 */
const VENV_PYTHON: string =
  process.env.PREFLIGHT_PYTHON ??
  path.join(
    CONFIG_DIR,
    ".venv",
    process.platform === "win32" ? "Scripts\\python.exe" : "bin/python",
  )

const MEMORY_SCRIPT     = path.join(CONFIG_DIR, "memory.py")
const TASKS_SCRIPT      = path.join(CONFIG_DIR, "tasks.py")
const CLASSIFIER_SCRIPT = path.join(CONFIG_DIR, "classifier.py")
const TRANSLATOR_SCRIPT = path.join(CONFIG_DIR, "translator.py")
const EXTRACTOR_SCRIPT  = path.join(CONFIG_DIR, "extractor.py")
const CONFIG_FILE       = path.join(CONFIG_DIR, "preflight.config.json")

// ─── Config (Feature 4) ───────────────────────────────────────────────────────

interface PreflightConfig {
  retrievalConfidenceThreshold: number
  useLLMClassifier: boolean
  anthropicApiKey: string | null
  interactiveMode: boolean
}

const DEFAULT_CONFIG: PreflightConfig = {
  retrievalConfidenceThreshold: 0.65,
  useLLMClassifier: true,
  anthropicApiKey: null,
  interactiveMode: true,
}

function loadConfig(): PreflightConfig {
  try {
    if (fs.existsSync(CONFIG_FILE)) {
      const raw = fs.readFileSync(CONFIG_FILE, "utf8")
      return { ...DEFAULT_CONFIG, ...(JSON.parse(raw) as Partial<PreflightConfig>) }
    }
  } catch {
    // Fall back to defaults on parse error
  }
  return { ...DEFAULT_CONFIG }
}

// ─── Project ID (Feature 2) ───────────────────────────────────────────────────

/**
 * Derive a short, stable identifier for the current project.
 * Prefers the git repository root; falls back to cwd.
 */
function getProjectId(): string {
  try {
    const gitRoot = execSync("git rev-parse --show-toplevel", {
      encoding: "utf8",
      cwd: process.cwd(),
      timeout: 2000,
    }).trim()
    return hashString(gitRoot)
  } catch {
    return hashString(process.cwd())
  }
}

function hashString(s: string): string {
  let h = 0
  for (let i = 0; i < s.length; i++) {
    h = ((h << 5) - h) + s.charCodeAt(i)
    h |= 0
  }
  return Math.abs(h).toString(16).slice(0, 8)
}

// ─── Async Python bridge (Feature 1 Bug 2) ───────────────────────────────────

/**
 * Invoke the venv Python interpreter with the given args and return stdout.
 * Replaces all previous spawnSync calls with a proper async version.
 */
function callPython(args: string[], timeoutMs: number = 30_000): Promise<string> {
  return new Promise((resolve, reject) => {
    const proc = spawn(VENV_PYTHON, args, { encoding: "utf8" } as never)
    let stdout = ""
    let stderr = ""
    let killed = false

    const timer = setTimeout(() => {
      killed = true
      proc.kill()
      reject(new Error("timeout"))
    }, timeoutMs)

    proc.stdout?.on("data", (d: string) => { stdout += d })
    proc.stderr?.on("data", (d: string) => { stderr += d })

    proc.on("close", (code: number) => {
      clearTimeout(timer)
      if (killed) return
      if (code !== 0) reject(new Error(`exit ${code}: ${stderr}`))
      else resolve(stdout)
    })
  })
}

// ─── Deduplication helpers ────────────────────────────────────────────────────

async function isDuplicate(key: string): Promise<boolean> {
  try {
    const stdout = await callPython([MEMORY_SCRIPT, "check_dedup", key], 5_000)
    return stdout.trim() === "EXISTS"
  } catch { return false }
}

async function markStored(key: string): Promise<void> {
  try {
    await callPython([MEMORY_SCRIPT, "mark_stored", key], 5_000)
  } catch { /* silently ignore */ }
}

// ─── Memory functions ─────────────────────────────────────────────────────────

/**
 * Store a message as a fact only if it has not been seen before
 * (deduplication via SHA-256 prefix key).
 */
async function storeSessionMessage(
  projectId: string,
  sessionId: string,
  text: string,
): Promise<void> {
  const key = crypto.createHash("sha256").update(text).digest("hex").slice(0, 16)
  if (await isDuplicate(key)) return
  try {
    await callPython([MEMORY_SCRIPT, "store_fact", projectId, sessionId, text], 15_000)
    await markStored(key)
  } catch { /* silently ignore */ }
}

async function storeMemory(
  projectId: string,
  sessionId: string,
  text: string,
): Promise<void> {
  try {
    await callPython([MEMORY_SCRIPT, "store_fact", projectId, sessionId, text], 15_000)
  } catch { /* silently ignore */ }
}

async function retrieveMemory(
  projectId: string,
  sessionId: string,
  prompt: string,
  threshold: number,
): Promise<string[]> {
  try {
    const stdout = await callPython(
      [MEMORY_SCRIPT, "retrieve_facts", projectId, sessionId, prompt, "3", String(threshold)],
      15_000,
    )
    return (JSON.parse(stdout.trim()) as string[]) ?? []
  } catch { return [] }
}

// ─── Task snapshot functions ──────────────────────────────────────────────────

async function retrieveSimilarTasks(
  projectId: string,
  sessionId: string,
  prompt: string,
  threshold: number,
): Promise<Array<{ type: string; prompt: string; summary: string; score: number }>> {
  try {
    const stdout = await callPython(
      [TASKS_SCRIPT, "retrieve_similar", projectId, sessionId, prompt, "3", String(threshold)],
      15_000,
    )
    return JSON.parse(stdout.trim()) ?? []
  } catch { return [] }
}

async function createTaskSnapshot(
  projectId: string,
  sessionId: string,
  taskType: string,
  origPrompt: string,
  summary: string,
): Promise<void> {
  try {
    await callPython(
      [TASKS_SCRIPT, "create_snapshot", projectId, sessionId, taskType, origPrompt, summary],
      15_000,
    )
  } catch { /* silently ignore */ }
}

// ─── Slot fill functions ──────────────────────────────────────────────────────

async function storeSlotFill(
  projectId: string,
  sessionId: string,
  slotName: string,
  value: string,
): Promise<void> {
  try {
    await callPython(
      [MEMORY_SCRIPT, "store_slot_fill", projectId, sessionId, slotName, value],
      5_000,
    )
  } catch { /* silently ignore */ }
}

async function retrieveSlotFills(
  projectId: string,
  sessionId: string,
): Promise<Array<{ session_id: string; slot_name: string; value: string }>> {
  try {
    const stdout = await callPython(
      [MEMORY_SCRIPT, "retrieve_slot_fills", projectId, sessionId],
      5_000,
    )
    return JSON.parse(stdout.trim()) ?? []
  } catch { return [] }
}

// ─── Classifier (Feature 5) ───────────────────────────────────────────────────

/**
 * Classify the task type using keyword scoring with LLM fallback (classifier.py).
 * Returns null if classification fails or is not confident enough.
 */
async function classify(
  prompt: string,
  useLLM: boolean,
  apiKey: string | null,
): Promise<string | null> {
  try {
    const payload = JSON.stringify({ prompt, useLLM, apiKey })
    const stdout  = await callPython([CLASSIFIER_SCRIPT, payload], 10_000)
    const result  = JSON.parse(stdout.trim()) as { type: string | null }
    return result.type ?? null
  } catch { return null }
}

// ─── LLM council: session fact extraction ───────────────────────────────────────

/**
 * Run session summary through extractor.py (keyword + optional LLM council).
 * Returns curated facts worth storing in long-term memory.
 */
async function extractFacts(
  text: string,
  apiKey: string | null,
): Promise<string[]> {
  try {
    const payload = JSON.stringify({ text, apiKey })
    const stdout  = await callPython([EXTRACTOR_SCRIPT, payload], 15_000)
    const result  = JSON.parse(stdout.trim()) as { facts?: string[] }
    return result.facts ?? []
  } catch { return [] }
}

// ─── Intent-to-technical translation (Feature 8) ─────────────────────────────

/** Slots whose values should be elicited via plain-language questions. */
const TECHNICAL_SLOTS: readonly string[] = [
  "stack", "language", "framework", "testing_framework", "database", "architecture_pattern",
]

/**
 * Plain-language questions — used when an API key is present so the answer
 * can be translated into a precise technical value (Feature 8).
 */
const INTENT_QUESTIONS: Record<string, string> = {
  testing_framework: "Do you want quick simple tests, or thorough tests with mocks and edge cases?",
  framework:         "Is this for a web UI, an API, or a command-line tool?",
  database:          "Does this need to handle a lot of relationships between data, or is it mostly storing records?",
}

/**
 * Direct technical questions — used in zero-config mode (no API key) or when
 * translation fails despite a key being present (Feature 8 graceful degrade).
 */
const DIRECT_QUESTIONS: Record<string, string> = {
  testing_framework: "What testing framework would you like to use (e.g. pytest, unittest, jest)?",
  framework:         "What framework would you like to use (e.g. FastAPI, Express, Django)?",
  database:          "What database would you like to use (e.g. PostgreSQL, SQLite, MongoDB)?",
}

/** Translate a plain-language answer into a concrete technical value via translator.py.
 * Returns null when no API key is available or when translation fails — caller falls
 * back to asking a direct technical question.
 */
async function translateSlot(
  slotName: string,
  userAnswer: string,
  context: string[],
  apiKey: string,
): Promise<string | null> {
  try {
    const payload = JSON.stringify({ slot: slotName, answer: userAnswer, context, apiKey })
    const stdout  = await callPython([TRANSLATOR_SCRIPT, payload], 10_000)
    const result  = JSON.parse(stdout.trim()) as { value: string | null }
    return result.value ?? null   // null = translation unavailable; caller asks direct question
  } catch { return null }
}

// ─── Slot filling with interactive confirmation (Feature 6) ──────────────────

interface SlotContext {
  [slotName: string]: string
}

/**
 * Load slot fills from the current session, ask the user to confirm or correct
 * each pre-filled value (Feature 6), and optionally translate plain-language
 * answers for technical slots (Feature 8).
 *
 * @param sendMessage  Function that sends a message and returns the user's reply.
 *                     Return undefined if no interactive reply is available.
 */
async function fillSlots(
  projectId: string,
  sessionId: string,
  _prompt: string,
  sendMessage: (text: string) => Promise<string | undefined>,
  config: PreflightConfig,
): Promise<SlotContext> {
  const fills = await retrieveSlotFills(projectId, sessionId)
  const slots: SlotContext = {}

  // Seed from stored fills
  for (const fill of fills) {
    slots[fill.slot_name] = fill.value
  }

  if (!config.interactiveMode) {
    return slots
  }

  // Feature 6: confirm or correct each pre-filled slot
  for (const fill of fills) {
    const reply = await sendMessage(
      `[pre-filled] ${fill.slot_name}: ${fill.value} (from previous session)\n` +
      `Reply "ok" to accept, or type your correction:`,
    )
    if (reply && reply.trim().toLowerCase() !== "ok" && reply.trim() !== "") {
      const corrected = reply.trim()
      slots[fill.slot_name] = corrected
      await storeSlotFill(projectId, sessionId, fill.slot_name, corrected)
    }
  }

  // Feature 8: elicit missing technical slots.
  // With API key   → ask plain-language question, translate, confirm.
  // Without API key → ask direct technical question (zero-config mode).
  // Translation null (API failure) → fall back to direct question.
  if (config.interactiveMode) {
    const hasApiKey    = !!config.anthropicApiKey
    const knownContext = Object.values(slots)

    for (const slot of TECHNICAL_SLOTS) {
      if (slots[slot] !== undefined) continue   // already known

      let finalValue: string | undefined

      if (hasApiKey && INTENT_QUESTIONS[slot]) {
        // Ask plain-language question and attempt translation
        const answer = await sendMessage(INTENT_QUESTIONS[slot])
        if (!answer || !answer.trim()) continue

        const technical = await translateSlot(
          slot, answer.trim(), knownContext, config.anthropicApiKey!,
        )

        if (technical !== null) {
          // Translation succeeded — show for confirmation before using
          const confirm = await sendMessage(
            `Interpreted "${answer.trim()}" as: ${technical} for ${slot}\n` +
            `Reply "ok" to accept, or type the exact value:`,
          )
          finalValue =
            confirm && confirm.trim().toLowerCase() !== "ok" && confirm.trim() !== ""
              ? confirm.trim()
              : technical
        } else {
          // Translation failed despite having a key — fall back to direct question
          const directQ = DIRECT_QUESTIONS[slot] ?? `What ${slot} do you want to use?`
          const directAnswer = await sendMessage(directQ)
          if (!directAnswer || !directAnswer.trim()) continue
          finalValue = directAnswer.trim()
        }
      } else if (DIRECT_QUESTIONS[slot] !== undefined) {
        // Zero-config mode: no API key — ask technical question directly
        const directAnswer = await sendMessage(
          DIRECT_QUESTIONS[slot] ?? `What ${slot} do you want to use?`,
        )
        if (!directAnswer || !directAnswer.trim()) continue
        finalValue = directAnswer.trim()
      }
      // Slots with no question defined in either map are silently skipped

      if (finalValue) {
        slots[slot] = finalValue
        knownContext.push(finalValue)
        await storeSlotFill(projectId, sessionId, slot, finalValue)
      }
    }
  }

  return slots
}

// ─── Prompt enrichment ────────────────────────────────────────────────────────

function buildEnrichedPrompt(
  originalPrompt: string,
  taskType: string | null,
  similarTasks: Array<{ type: string; prompt: string; summary: string; score: number }>,
  memories: string[],
  slots: SlotContext,
): string {
  const lines: string[] = []

  if (memories.length > 0) {
    lines.push("## Relevant context from memory")
    for (const mem of memories) lines.push(`- ${mem}`)
    lines.push("")
  }

  if (similarTasks.length > 0) {
    lines.push("## Similar past tasks")
    for (const task of similarTasks) {
      lines.push(`- [${task.type}] ${task.summary}`)
    }
    lines.push("")
  }

  const filledSlots = Object.entries(slots)
  if (filledSlots.length > 0) {
    lines.push("## Project context")
    for (const [key, value] of filledSlots) lines.push(`- ${key}: ${value}`)
    lines.push("")
  }

  if (taskType) {
    lines.push(`## Task classification: ${taskType}`)
    lines.push("")
  }

  lines.push("## Request")
  lines.push(originalPrompt)

  return lines.join("\n")
}

// ─── Plugin entry point ───────────────────────────────────────────────────────

export const PreflightPlugin: Plugin = async () => {
  const config       = loadConfig()
  const projectId    = getProjectId()
  // #1: only enrich the first prompt of each session — follow-ups already have
  // the context in the conversation window, no need to re-inject every time.
  const seenSessions = new Set<string>()

  return {
    // ── Enrich every prompt before it reaches the LLM ──────────────────────
    "tui.prompt.append": async (
      input: { sessionID: string },
      output: { text: string },
    ) => {
      const sessionId      = input.sessionID
      const originalPrompt = output.text

      // Store prompt in background regardless (always track what was asked)
      storeSessionMessage(projectId, sessionId, originalPrompt).catch(() => {})

      // #1: skip enrichment on follow-up messages — context already in window
      if (seenSessions.has(sessionId)) return
      seenSessions.add(sessionId)

      const [taskType, similarTasks, memories, fills] = await Promise.all([
        classify(originalPrompt, config.useLLMClassifier, config.anthropicApiKey),
        retrieveSimilarTasks(projectId, sessionId, originalPrompt, config.retrievalConfidenceThreshold),
        retrieveMemory(projectId, sessionId, originalPrompt, config.retrievalConfidenceThreshold),
        retrieveSlotFills(projectId, sessionId),
      ])

      const slots: SlotContext = {}
      for (const f of fills) slots[f.slot_name] = f.value

      output.text = buildEnrichedPrompt(originalPrompt, taskType, similarTasks, memories, slots)
    },

    // ── Save task snapshots when session goes idle (task likely completed) ──
    "session.idle": async (input: { sessionID: string; lastMessage?: string }) => {
      if (!input.lastMessage) return

      // #4: LLM council — extract curated facts before storing anything
      const [taskType, facts] = await Promise.all([
        classify(input.lastMessage, false, null),
        extractFacts(input.lastMessage, config.anthropicApiKey),
      ])

      // Store each council-approved fact in long-term memory
      for (const fact of facts) {
        storeMemory(projectId, input.sessionID, fact).catch(() => {})
      }

      // Also snapshot the task type + summary for weighted task retrieval
      if (taskType) {
        const summary = facts.length > 0 ? facts[0] : input.lastMessage
        await createTaskSnapshot(projectId, input.sessionID, taskType, "", summary)
      }
    },

    // ── Custom tools so Claude can persist what it learns ───────────────────
    tool: {
      preflight_store_slot: tool({
        description:
          "Store a project slot value (e.g. framework, database, testing_framework) " +
          "so it is remembered across sessions. Call this after the user answers a " +
          "question about their project setup.",
        args: {
          session_id: tool.schema.string(),
          slot_name:  tool.schema.string(),
          value:      tool.schema.string(),
        },
        async execute(args) {
          await storeSlotFill(projectId, args.session_id, args.slot_name, args.value)
          return `Stored ${args.slot_name} = ${args.value}`
        },
      }),

      preflight_store_memory: tool({
        description:
          "Store a fact or insight about the current project so it can be retrieved " +
          "in future sessions. Use this when the user states something important about " +
          "the codebase, architecture, or workflow.",
        args: {
          session_id: tool.schema.string(),
          fact:       tool.schema.string(),
        },
        async execute(args) {
          await storeMemory(projectId, args.session_id, args.fact)
          return `Stored memory: ${args.fact}`
        },
      }),
    },
  }
}

  plugin.on("chat.message", async (msg: unknown) => {
    const message = msg as ChatMessage
    const { sessionId, prompt } = message

    try {
      // Classify the task type (Feature 5)
      const taskType = await classify(prompt, config.useLLMClassifier, config.anthropicApiKey)

      // Retrieve relevant context in parallel
      const [similarTasks, memories] = await Promise.all([
        retrieveSimilarTasks(projectId, sessionId, prompt, config.retrievalConfidenceThreshold),
        retrieveMemory(projectId, sessionId, prompt, config.retrievalConfidenceThreshold),
      ])

      // Fill slots with interactive confirmation (Features 6 + 8)
      const slots = await fillSlots(
        projectId, sessionId, prompt, message.reply.bind(message), config,
      )

      // Build enriched prompt and pass it on
      const enriched = buildEnrichedPrompt(prompt, taskType, similarTasks, memories, slots)
      message.setPrompt(enriched)

      // Persist message for future retrieval (async, non-blocking for the chat)
      void storeSessionMessage(projectId, sessionId, prompt)

      // Create a task snapshot if we have a classification
      if (taskType) {
        void createTaskSnapshot(
          projectId, sessionId, taskType, prompt, prompt.slice(0, 200),
        )
      }
    } catch (err) {
      // Never break the chat on enrichment failure — pass original prompt through
      console.error("[preflight] enrichment error:", err)
    }
  })
}
