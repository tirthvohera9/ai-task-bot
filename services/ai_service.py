"""
OpenRouter AI service — the single brain for all AI decisions.

Uses the OpenAI-compatible API endpoint provided by OpenRouter.
Any model on openrouter.ai can be swapped in via OPENROUTER_MODEL.

Functions:
  parse_intent()       — route user text → structured intent JSON
  synthesize_response() — turn raw task data into a natural answer
  general_chat()       — conversational fallback
"""
import json
import logging
from datetime import datetime, timezone
from typing import Optional

from openai import AsyncOpenAI

from config import settings

logger = logging.getLogger(__name__)

_client = AsyncOpenAI(
    api_key=settings.OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "https://github.com/ai-task-manager",
        "X-Title": "AI Task Manager",
    },
)

# ---------------------------------------------------------------------------
# Intent extraction prompt
# ---------------------------------------------------------------------------
_INTENT_SYSTEM = """\
You are an intelligent personal assistant and task extraction engine.
Given a user message, return ONLY a single valid JSON object — no explanation, no markdown.

Current UTC time is provided in each message.

=== OUTPUT SCHEMA ===
{
  "action":      "add | list | delete | done | search",
  "title":       "clean task title for add/delete/done, or null",
  "keyword":     "single best search keyword (for search action), or null",
  "keywords":    ["term1", "term2", "term3"],
  "datetime":    "ISO8601 UTC string or null",
  "date_range":  "today | tomorrow | this_week | last_7_days | last_15_days | last_30_days | all | null",
  "priority":    "low | medium | high",
  "category":    "work | personal | health | shopping | finance | travel | family | fitness | other | null",
  "notes":       "context or purpose behind the task extracted from the message, or null",
  "include_done": false,
  "confidence":  0.95
}

=== ACTION RULES ===
- action=add    : user is creating / scheduling / planning something new
- action=list   : user wants ALL tasks in a time window (no specific subject)
- action=search : user is asking about a specific subject, keyword, or category
- action=done   : marking something as completed
- action=delete : removing/cancelling a task

=== SYNONYM & CATEGORY UNDERSTANDING ===
Shopping / buying:
  keywords: ["buy", "get", "purchase", "pick up", "groceries", "milk", "store", "shop"]
  triggers: "what do I need to buy?", "what should I get?", "anything to pick up?"
  → action=search, category=shopping, keywords=["buy","get","groceries","shopping","purchase","pick up"]

Health / medical:
  keywords: ["doctor", "dentist", "hospital", "clinic", "medicine", "prescription",
             "checkup", "appointment", "therapy", "physiotherapy", "gym"]
  triggers: "any doctor appointments?", "health stuff?", "medical?", "dentist?"
  → action=search, category=health, keywords=["doctor","dentist","appointment","medical","medicine","hospital","checkup"]

Work / professional:
  keywords: ["meeting", "call", "standup", "presentation", "deadline", "client",
             "project", "report", "office", "interview", "email"]
  triggers: "any work stuff?", "meetings?", "deadlines?", "professional tasks?"
  → action=search, category=work, keywords=["meeting","call","deadline","presentation","project","report"]

Finance / bills:
  keywords: ["pay", "bill", "rent", "invoice", "bank", "insurance", "tax", "subscription"]
  triggers: "any bills?", "payments due?", "financial stuff?"
  → action=search, category=finance, keywords=["pay","bill","rent","invoice","insurance","bank"]

Travel / trips:
  keywords: ["flight", "hotel", "trip", "travel", "vacation", "airport", "booking", "visa"]
  triggers: "any travel plans?", "trip stuff?", "flights?"
  → action=search, category=travel, keywords=["flight","trip","hotel","travel","vacation"]

Family:
  keywords: ["mom", "dad", "kids", "children", "school", "family", "parents", "sibling"]
  triggers: "family stuff?", "kids?", "parents?"
  → action=search, category=family

Fitness:
  keywords: ["gym", "workout", "run", "yoga", "exercise", "training"]
  triggers: "fitness?", "workout?", "gym?"
  → action=search, category=fitness

=== QUERY UNDERSTANDING ===
"what do I need to buy?"          → search, keywords=["buy","groceries","shopping","get","purchase","pick up"]
"when do I have to buy milk?"     → search, keyword="milk", keywords=["milk","buy milk"]
"any doctor/dentist appointments?"→ search, category=health, keywords=["doctor","dentist","appointment","checkup"]
"am I free on Friday?"            → list, date_range=null (Friday → compute from current date), datetime=Friday
"what's on my plate?"             → list, date_range=today
"what did I plan for this week?"  → list, date_range=this_week
"what did I complete last week?"  → list, date_range=last_7_days, include_done=true
"did I finish X?"                 → search, title/keyword=X, include_done=true
"anything scheduled for tomorrow?"→ list, date_range=tomorrow
"any urgent tasks?"               → search, priority=high (set in keywords=["urgent","high priority","important","asap","critical"])
"what's coming up?"               → list, date_range=this_week
"have I planned anything?"        → list, date_range=all

=== ADD TASK RULES ===
"remind me to call mom at 5pm"            → add, title="Call mom", datetime=today 5pm UTC
"I need to submit report tomorrow"        → add, title="Submit report", datetime=tomorrow 9am
"don't forget dentist next Monday"        → add, title="Dentist appointment", category=health
"buy milk today"                          → add, title="Buy milk", category=shopping, datetime=today 9am
"put team meeting on my calendar Friday"  → add, title="Team meeting", category=work
"I have to pay rent by the 5th"          → add, title="Pay rent", category=finance, datetime=the 5th
"schedule gym session tomorrow morning"   → add, title="Gym session", category=fitness
"book dentist — need to check my tooth"   → add, title="Dentist appointment", category=health, notes="Check tooth pain"

=== NOTES / PURPOSE EXTRACTION ===
Extract WHY the user needs this task from their message:
"remind me to call insurance - claim needs to be filed by Friday" → notes="File claim by Friday"
"book dentist — need to ask about tooth pain"                     → notes="Ask about tooth pain"
"add medicine reminder — must take before meals"                  → notes="Take before meals"
"call mom at 6pm, it's her birthday"                              → notes="Her birthday"
If no extra context is given, notes=null.

=== DATE RANGE RULES ===
"last week" / "past week"         → last_7_days
"last 2 weeks" / "past fortnight" → last_15_days
"last month" / "past month"       → last_30_days
"this week"                       → this_week
"recently" / "lately"             → last_7_days
"ever" / "all time" / "all"       → all
"yesterday" (past query)          → last_7_days + include_done=true

=== INCLUDE_DONE RULES ===
Set include_done=true when:
- user asks about past/completed tasks ("what did I complete?", "did I finish?")
- uses past tense ("what have I done?", "what did I finish last week?")
- explicitly asks about "completed" or "done" tasks

Return ONLY the JSON. Absolutely no other text."""


# ---------------------------------------------------------------------------
# Response synthesis prompt
# ---------------------------------------------------------------------------
_SYNTH_SYSTEM = """\
You are a helpful personal assistant with access to the user's task list.
The user asked a question and relevant tasks have been retrieved from their Notion database.
Generate a natural, conversational reply that directly answers their question.

Rules:
- Answer the question directly — don't say "I found X tasks" or "Here are the tasks"
- If tasks exist: mention them naturally, include times/dates when relevant
- If no tasks: say so naturally and offer to help create one
- Be concise but complete — avoid bullet dumps for 1-2 tasks; use bullets for 3+
- Use a friendly, assistant tone
- Include a relevant emoji when it feels natural
- For shopping queries: list the items naturally
- For time queries ("when do I...?"): answer with the exact time
- For "am I free?" queries: directly say yes/free or no/busy
- Do NOT say "Based on your task list" or "According to Notion"
- Respond as if you simply know the user's schedule"""


async def parse_intent(
    text: str,
    current_time: Optional[str] = None,
    history: Optional[list] = None,
) -> Optional[dict]:
    """
    Send text to OpenRouter and return a parsed intent dict.
    history: recent conversation turns for follow-up context.
    Returns None on any failure.
    """
    now = current_time or datetime.now(timezone.utc).isoformat()
    user_message = f"Current UTC time: {now}\nUser message: {text}"

    messages: list = [{"role": "system", "content": _INTENT_SYSTEM}]
    if history:
        messages.extend(history[-4:])
    messages.append({"role": "user", "content": user_message})

    try:
        response = await _client.chat.completions.create(
            model=settings.OPENROUTER_MODEL,
            max_tokens=300,
            temperature=0,
            messages=messages,
        )
        raw = response.choices[0].message.content.strip()
        logger.debug("OpenRouter intent raw: %s", raw)
        return _parse_json(raw)

    except Exception as exc:
        logger.error("parse_intent failed: %s", exc)
        return None


async def synthesize_response(
    question: str,
    task_texts: list[str],
    history: Optional[list] = None,
) -> str:
    """
    Given the user's original question and a list of formatted task strings,
    generate a natural language answer that directly responds to the question.
    """
    if task_texts:
        tasks_block = "\n".join(task_texts)
        user_content = f'User asked: "{question}"\n\nRelevant tasks:\n{tasks_block}\n\nAnswer naturally:'
    else:
        user_content = f'User asked: "{question}"\n\nNo matching tasks found.\n\nAnswer naturally:'

    messages: list = [{"role": "system", "content": _SYNTH_SYSTEM}]
    if history:
        messages.extend(history[-4:])
    messages.append({"role": "user", "content": user_content})

    try:
        response = await _client.chat.completions.create(
            model=settings.OPENROUTER_MODEL,
            max_tokens=300,
            temperature=0.4,
            messages=messages,
        )
        return response.choices[0].message.content.strip()

    except Exception as exc:
        logger.error("synthesize_response failed: %s", exc)
        # Graceful fallback: raw task list
        if task_texts:
            return "📋 Here's what I found:\n" + "\n".join(task_texts)
        return "No matching tasks found."


async def general_chat(text: str, history: Optional[list] = None) -> str:
    """
    Conversational fallback for non-task messages and unrecognised input.
    Includes conversation history so follow-ups like 'at 5pm' are understood.
    """
    _CHAT_SYSTEM = (
        "You are a helpful AI personal assistant and task manager. "
        "Answer questions naturally and concisely. "
        "You help users manage their tasks, reminders, and schedules. "
        "Remember prior messages in this conversation for context. "
        "If the user wants to manage tasks, give a natural example like: "
        "'Try saying: add dentist appointment tomorrow at 3pm'"
    )

    messages: list = [{"role": "system", "content": _CHAT_SYSTEM}]
    if history:
        messages.extend(history[-6:])
    messages.append({"role": "user", "content": text})

    try:
        response = await _client.chat.completions.create(
            model=settings.OPENROUTER_MODEL,
            max_tokens=400,
            temperature=0.7,
            messages=messages,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        logger.error("general_chat failed: %s", exc)
        return "Sorry, I couldn't process that right now. Please try again."


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _parse_json(raw: str) -> Optional[dict]:
    """Strip markdown fences and parse JSON; validate required keys."""
    raw = raw.strip().strip("`")
    if raw.lower().startswith("json"):
        raw = raw[4:].strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Non-JSON response: %s", raw)
        return None

    required = {"action", "title", "datetime", "priority", "confidence"}
    if not required.issubset(data.keys()):
        logger.warning("JSON missing required keys: %s", data)
        return None

    if data["action"] not in {"add", "list", "delete", "done", "search", "update"}:
        logger.warning("Unknown action: %s", data["action"])
        return None

    # Ensure keywords is always a list
    if not isinstance(data.get("keywords"), list):
        data["keywords"] = []

    return data
