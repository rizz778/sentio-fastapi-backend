from typing import Any, Dict, List, Optional

EMPATHY_INSTRUCTION = (
    "Whenever you receive a user's current emotion, always acknowledge it explicitly in your reply, "
    "mention the emotion, and respond with empathy and support tailored to that emotion. Never ignore the user's emotion."
)


def format_user_profile(user_profile: Dict[str, Any]) -> str:
    if not user_profile:
        return ""

    triggers = ", ".join(user_profile.get("triggers", []))
    coping = ", ".join(user_profile.get("copingMechanisms", []))
    preferences = ", ".join(user_profile.get("preferences", []))

    return (
        f"The user's name is {user_profile.get('name', 'Unknown')}. "
        f"Age: {user_profile.get('age', 'Unknown')}, Gender: {user_profile.get('gender', 'Unknown')}. "
        f"Stress Frequency: {user_profile.get('stressFrequency', 'Unknown')}. "
        f"Sleep: {user_profile.get('sleep', 'Unknown')}. "
        f"Exercise: {user_profile.get('exercise', 'Unknown')}. "
        f"Triggers: {triggers}. "
        f"Coping Mechanisms: {coping}. "
        f"Support System: {user_profile.get('supportSystem', 'Unknown')}. "
        f"Preferred Language: {user_profile.get('preferredLanguage', 'Unknown')}. "
        f"Response Style: {user_profile.get('responseStyle', 'Unknown')}. "
        f"Preferences: {preferences}. "
        "You MUST use this information when answering questions about the user. "
        "If the user asks for their name, always use the name from the profile above. "
        "If the user asks about themselves, always refer to the profile above. "
        "Never say you don't have access to the user's information. "
        "Don't use asterisks for emphasis, use all caps instead. "
        f"{EMPATHY_INSTRUCTION}"
    )


def format_conversation_history(conversation_history: List[Dict[str, Any]], limit: int = 5) -> str:
    if not conversation_history or not isinstance(conversation_history, list):
        return ""

    last_msgs = conversation_history[-limit:] if len(conversation_history) > limit else conversation_history
    history_entries = [
        f"[{msg.get('sender', '')}: {msg.get('text', '')}]"
        for msg in last_msgs
        if isinstance(msg, dict)
    ]
    if not history_entries:
        return ""
    return "Conversation so far: " + " ".join(history_entries)


def build_chat_system_instruction(
    user_profile: Optional[Dict[str, Any]] = None,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
    custom_instruction: Optional[str] = None,
) -> str:
    profile_str = format_user_profile(user_profile) if user_profile else ""
    history_str = format_conversation_history(conversation_history) if conversation_history else ""

    parts = [p for p in [profile_str, history_str, custom_instruction] if p]
    system_instruction = " ".join(parts).strip()

    if EMPATHY_INSTRUCTION not in system_instruction:
        system_instruction = f"{system_instruction}\n{EMPATHY_INSTRUCTION}".strip()

    return system_instruction or EMPATHY_INSTRUCTION
