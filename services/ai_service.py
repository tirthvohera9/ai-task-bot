"""
OpenRouter AI service — the single brain for all AI decisions.

Functions:
  parse_intent()        → structured intent JSON from user text
  synthesize_response() → natural answer from raw task data
  general_chat()        → conversational fallback
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

=== OUTPUT SCHEMA ===
{
  "action":       "add | list | delete | done | search | update",
  "title":        "task title for add/delete/done/update, or null",
  "keyword":      "primary search keyword (for search), or null",
  "keywords":     ["term1", "term2"],
  "update_field": "datetime | priority | title | status | null",
  "update_value": "new value for the updated field, or null",
  "datetime":     "ISO8601 UTC string or null",
  "date_range":   "today | tomorrow | this_week | last_7_days | last_15_days | last_30_days | all | null",
  "priority":     "low | medium | high",
  "category":     "work | personal | health | shopping | finance | travel | family | fitness | other | null",
  "notes":        "purpose/context extracted from the message, or null",
  "recurrence":   "none | daily | weekly:MON | weekly:TUE | weekly:WED | weekly:THU | weekly:FRI | weekly:SAT | weekly:SUN | monthly:1..31 | null",
  "include_done": false,
  "confidence":   0.95
}

=== ACTION RULES ===
- add    : creating / scheduling / planning something new
- list   : wants ALL tasks in a time window (no specific subject)
- search : asking about a specific subject/keyword/category
- done   : marking something as completed
- delete : removing/cancelling a task
- update : changing an existing task's field (datetime, priority, title, or status)

=== UPDATE EXAMPLES ===
"move meeting to 6pm"         → action=update, title="meeting",  update_field="datetime", update_value="6pm"
"change dentist to next week" → action=update, title="dentist",  update_field="datetime", update_value="next week"
"mark gym as high priority"   → action=update, title="gym",      update_field="priority", update_value="high"
"rename report to final draft"→ action=update, title="report",   update_field="title",    update_value="final draft"
"reschedule rent to the 10th" → action=update, title="rent",     update_field="datetime", update_value="the 10th"

=== RECURRENCE RULES ===
"every day" / "daily"              → recurrence="daily"
"every Monday"                     → recurrence="weekly:MON"
"every week on Friday"             → recurrence="weekly:FRI"
"every month on the 5th"           → recurrence="monthly:5"
No recurrence mentioned            → recurrence="none"

=== SYNONYM UNDERSTANDING ===
Shopping/buying: "buy", "get", "purchase", "pick up", "groceries", "store"
  → category=shopping, keywords=["buy","get","groceries","shopping","purchase","pick up"]

Health/medical: "doctor", "dentist", "hospital", "clinic", "medicine", "prescription",
                "checkup", "appointment", "therapy"
  → category=health, keywords=["doctor","dentist","appointment","medical","medicine","hospital"]

Work: "meeting", "call", "standup", "presentation", "deadline", "client", "project",
      "report", "office", "interview", "email"
  → category=work

Finance: "pay", "bill", "rent", "invoice", "bank", "insurance", "tax", "subscription"
  → category=finance

Travel: "flight", "hotel", "trip", "travel", "vacation", "airport", "booking", "visa"
  → category=travel

Family: "mom", "dad", "kids", "children", "school", "family", "parents"
  → category=family

Fitness: "gym", "workout", "run", "yoga", "exercise", "training"
  → category=fitness

=== NOTES/PURPOSE EXTRACTION ===
"book dentist — need to ask about tooth pain"   → notes="Ask about tooth pain"
"remind me to call insurance, claim by Friday"  → notes="File claim by Friday"
"add medicine at 8am — take before meals"       → notes="Take before meals"

=== DATE RANGE RULES ===
"last week" / "past week"   → last_7_days
"last 2 weeks" / "fortnight"→ last_15_days
"last month"                → last_30_days
"this week"                 → this_week
"recently"                  → last_7_days

=== INCLUDE_DONE ===
Set include_done=true when user asks about completed, past, or finished tasks.

Return ONLY the JSON. No other text whatsoever."""


# ---------------------------------------------------------------------------
# Response synthesis prompt
# ---------------------------------------------------------------------------
_SYNTH_SYSTEM = """\
You are a concise personal assistant answering questions about the user's tasks.

Rules:
- Get to the point immediately — no preamble
- 1–2 tasks: answer in one sentence with time/date
- 3+ tasks: short numbered list
- No tasks found: say so in one sentence, offer to add one
- Never say "I found", "According to", "Based on the data"
- Time queries ("when do I…?"): give exact time and day only
- Yes/no queries ("am I free?"): lead with yes or no"""


async def parse_intent(
    text: str,
    current_time: Optional[str] = None,
    history: Optional[list] = None,
    user_tz: str = "UTC",
) -> Optional[dict]:
    """Parse user text into a structured intent dict via OpenRouter."""
    now = current_time or datetime.now(timezone.utc).isoformat()
    user_message = f"Current UTC time: {now}\nUser timezone: {user_tz}\nUser message: {text}"

    messages: list = [{"role": "system", "content": _INTENT_SYSTEM}]
    if history:
        messages.extend(history[-4:])
    messages.append({"role": "user", "content": user_message})

    try:
        response = await _client.chat.completions.create(
            model=settings.OPENROUTER_MODEL,
            max_tokens=350,
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
    """Turn a list of task strings into a natural answer for the user's question."""
    if task_texts:
        tasks_block = "\n".join(task_texts)
        user_content = f'Question: "{question}"\n\nTask data:\n{tasks_block}\n\nAnswer:'
    else:
        user_content = f'Question: "{question}"\n\nNo matching tasks found.\n\nAnswer:'

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
        return ("📋 " + "\n".join(task_texts)) if task_texts else "No matching tasks found."


async def general_chat(
    text: str,
    history: Optional[list] = None,
    task_context: Optional[str] = None,
    user_profile: Optional[str] = None,
    chat_streak: int = 0,
) -> str:
    """
    Task-aware conversational reply.

    Behaviour changes by streak:
      0–2 turns : helpful, task-aware, suggest actions when relevant
      3+  turns : brief drift-back — acknowledge then pivot to action
    """
    task_block    = f"\nUser's tasks right now: {task_context}" if task_context else ""
    profile_block = f"\nUser profile: {user_profile}" if user_profile else ""

    if chat_streak >= 3:
        system = (
            "You are a concise task-focused assistant. "
            "The user has been chatting for a while without taking action. "
            "Respond in 1–2 sentences, then redirect them toward doing something concrete. "
            "Be direct, not preachy."
            f"{task_block}{profile_block}"
        )
        max_tokens = 120
    else:
        system = (
            "You are a personal assistant who knows the user's schedule. "
            "Be brief and direct — max 3 sentences. "
            "If the user mentions something they need to do, offer to add it as a task. "
            "If they seem overwhelmed, reference their task count and offer to help prioritize. "
            "If they ask a question you can answer quickly, just answer it. "
            "Never use filler phrases like 'Great question!' or 'Of course!'."
            f"{task_block}{profile_block}"
        )
        max_tokens = 250

    messages: list = [{"role": "system", "content": system}]
    if history:
        messages.extend(history[-4:])
    messages.append({"role": "user", "content": text})

    try:
        response = await _client.chat.completions.create(
            model=settings.OPENROUTER_MODEL,
            max_tokens=max_tokens,
            temperature=0.6,
            messages=messages,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        logger.error("general_chat failed: %s", exc)
        return "Got it. What do you want to do about it?"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _parse_json(raw: str) -> Optional[dict]:
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

    if not isinstance(data.get("keywords"), list):
        data["keywords"] = []

    return data
