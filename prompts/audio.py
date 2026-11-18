import json
from typing import Any, Dict, List, Optional

DEFAULT_THERAPIST_INSTRUCTION = "You are a Therapist"


def format_audio_user_profile(user_profile: Dict[str, Any]) -> str:
    if not user_profile:
        return ""
    return f"User profile: {json.dumps(user_profile)}. "


def format_audio_history(conversation_history: List[Dict[str, Any]], limit: int = 5) -> str:
    if not conversation_history or not isinstance(conversation_history, list):
        return ""

    last_msgs = (
        conversation_history[-limit:]
        if len(conversation_history) > limit
        else conversation_history
    )
    entries = [
        f"[{msg.get('sender', '')}: {msg.get('text', '')} ]"
        for msg in last_msgs
        if isinstance(msg, dict)
    ]
    if not entries:
        return ""
    return "Conversation so far: " + " ".join(entries)


def build_audio_system_instruction(
    user_profile: Optional[Dict[str, Any]] = None,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
    base_instruction: str = DEFAULT_THERAPIST_INSTRUCTION,
) -> str:
    instruction = base_instruction or DEFAULT_THERAPIST_INSTRUCTION

    if user_profile:
        instruction += f" {format_audio_user_profile(user_profile)}"

    if conversation_history:
        history_str = format_audio_history(conversation_history)
        if history_str:
            instruction += f" {history_str}"

    return instruction.strip()
