from typing import Dict, List


# ============================================================
# CONVERSATION MEMORY
# ============================================================

# In-memory conversation storage.
#
# Structure:
#
# {
#     "session_id": [
#         {
#             "role": "user",
#             "content": "..."
#         },
#         {
#             "role": "assistant",
#             "content": "..."
#         }
#     ]
# }
#
# Later we can replace this with Redis / SQLite / another
# persistent memory system.

_memory: Dict[str, List[dict]] = {}


# ============================================================
# SETTINGS
# ============================================================

# Maximum number of messages kept for one conversation.
MAX_MESSAGES = 12


# ============================================================
# GET MEMORY
# ============================================================

def get_memory(session_id: str) -> List[dict]:
    """
    Return conversation history for a session.
    """

    return _memory.get(
        session_id,
        []
    )


# ============================================================
# ADD USER MESSAGE
# ============================================================

def add_user_message(
    session_id: str,
    message: str
):
    """
    Store a user message.
    """

    if session_id not in _memory:
        _memory[session_id] = []

    _memory[session_id].append(
        {
            "role": "user",
            "content": message
        }
    )

    _trim_memory(session_id)


# ============================================================
# ADD ASSISTANT MESSAGE
# ============================================================

def add_assistant_message(
    session_id: str,
    message: str
):
    """
    Store Jojo's response.
    """

    if session_id not in _memory:
        _memory[session_id] = []

    _memory[session_id].append(
        {
            "role": "assistant",
            "content": message
        }
    )

    _trim_memory(session_id)


# ============================================================
# ADD COMPLETE EXCHANGE
# ============================================================

def add_exchange(
    session_id: str,
    user_message: str,
    assistant_message: str
):
    """
    Store one complete conversation exchange.
    """

    if session_id not in _memory:
        _memory[session_id] = []

    _memory[session_id].append(
        {
            "role": "user",
            "content": user_message
        }
    )

    _memory[session_id].append(
        {
            "role": "assistant",
            "content": assistant_message
        }
    )

    _trim_memory(session_id)


# ============================================================
# LIMIT MEMORY
# ============================================================

def _trim_memory(session_id: str):
    """
    Keep only the latest MAX_MESSAGES messages.
    """

    if session_id not in _memory:
        return

    if len(_memory[session_id]) > MAX_MESSAGES:

        _memory[session_id] = (
            _memory[session_id][-MAX_MESSAGES:]
        )


# ============================================================
# FORMAT MEMORY FOR GEMINI
# ============================================================

def format_memory(
    session_id: str
) -> str:
    """
    Convert conversation history into text that can later
    be supplied to Gemini.
    """

    history = get_memory(session_id)

    if not history:
        return "No previous conversation."

    lines = []

    for message in history:

        role = message["role"]
        content = message["content"]

        if role == "user":
            lines.append(
                f"User: {content}"
            )

        elif role == "assistant":
            lines.append(
                f"Jojo: {content}"
            )

    return "\n".join(lines)


# ============================================================
# CLEAR SESSION
# ============================================================

def clear_memory(session_id: str):
    """
    Delete conversation memory for one session.
    """

    _memory.pop(
        session_id,
        None
    )


# ============================================================
# DEBUG
# ============================================================

def memory_count(
    session_id: str
) -> int:
    """
    Return number of stored messages.
    """

    return len(
        get_memory(session_id)
    )