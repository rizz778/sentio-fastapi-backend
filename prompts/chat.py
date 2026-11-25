from typing import Any, Dict, List, Optional

EMPATHY_INSTRUCTION = (
    "Whenever you receive a user's current emotion, always acknowledge it explicitly in your reply, "
    "mention the emotion, and respond with empathy and support tailored to that emotion. Never ignore the user's emotion."
)


def format_user_profile(user_profile: Dict[str, Any]) -> str:
    if not user_profile or not isinstance(user_profile, dict):
        return ""

    # Unwrap if response is nested under "userProfile" key
    if "userProfile" in user_profile and isinstance(user_profile["userProfile"], dict):
        user_profile = user_profile["userProfile"]

    name = user_profile.get("name") or "Unknown"
    age = user_profile.get("age") or "Unknown"
    gender = user_profile.get("gender") or "Unknown"
    pref_lang = user_profile.get("preferredLanguage") or "Unknown"
    stress_freq = user_profile.get("stressFrequency") or "Unknown"
    sleep = user_profile.get("sleepHours") or user_profile.get("sleep") or "Unknown"
    exercise = user_profile.get("exerciseFrequency") or user_profile.get("exercise") or "Unknown"
    triggers = ", ".join(user_profile.get("triggers", [])) if isinstance(user_profile.get("triggers"), list) else ""
    coping = ", ".join(user_profile.get("copingMechanisms", [])) if isinstance(user_profile.get("copingMechanisms"), list) else ""
    support = user_profile.get("hasSupportSystem") if user_profile.get("hasSupportSystem") is not None else user_profile.get("supportSystem", "Unknown")
    response_style = user_profile.get("responsePreference") or user_profile.get("responseStyle") or "Unknown"
    preferences = ", ".join(user_profile.get("preferences", [])) if isinstance(user_profile.get("preferences"), list) else ""
    condition = user_profile.get("diagnosedCondition") or ""
    current_mood = user_profile.get("currentMood") or ""

    profile_parts = [
        f"The user's name is {name}.",
        f"Age: {age}, Gender: {gender}.",
        f"Preferred Language: {pref_lang}.",
        f"Stress Frequency: {stress_freq}.",
        f"Sleep Hours: {sleep}.",
        f"Exercise Frequency: {exercise}.",
        f"Triggers: {triggers or 'None noted'}.",
        f"Coping Mechanisms: {coping or 'None noted'}.",
        f"Support System: {support}.",
        f"Response Style: {response_style}.",
    ]
    if condition:
        profile_parts.append(f"Diagnosed Condition: {condition}.")
    if current_mood:
        profile_parts.append(f"Current Mood: {current_mood}.")
    if preferences:
        profile_parts.append(f"Preferences: {preferences}.")

    profile_parts.extend([
        "You MUST use this information when answering questions about the user.",
        "If the user asks for their name, always use the name from the profile above.",
        "If the user asks about themselves, always refer to the profile above.",
        "Never say you don't have access to the user's information.",
        "Don't use asterisks for emphasis, use all caps instead.",
        EMPATHY_INSTRUCTION,
    ])

    return " ".join(profile_parts)


def format_conversation_summary(summary: Optional[str]) -> str:
    if not summary or not summary.strip():
        return ""
    return f"[Ongoing Session Clinical Summary]:\n{summary.strip()}"


def format_conversation_history(conversation_history: List[Dict[str, Any]], limit: int = 5) -> str:
    if not conversation_history or not isinstance(conversation_history, list):
        return ""

    last_msgs = conversation_history[-limit:] if len(conversation_history) > limit else conversation_history
    history_entries = [
        f"[{msg.get('sender', '')}: {msg.get('text', '')}]"
        for msg in last_msgs
        if isinstance(msg, dict) and msg.get("text")
    ]
    if not history_entries:
        return ""
    return "[Recent Dialogue (Last 5 turns)]:\n" + "\n".join(history_entries)


def build_chat_system_instruction(
    user_profile: Optional[Dict[str, Any]] = None,
    conversation_summary: Optional[str] = None,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
    custom_instruction: Optional[str] = None,
) -> str:
    profile_str = format_user_profile(user_profile) if user_profile else ""
    summary_str = format_conversation_summary(conversation_summary)
    history_str = format_conversation_history(conversation_history, limit=5) if conversation_history else ""

    parts = [p for p in [profile_str, summary_str, history_str, custom_instruction] if p]
    system_instruction = "\n\n".join(parts).strip()

    if EMPATHY_INSTRUCTION not in system_instruction:
        system_instruction = f"{system_instruction}\n\n{EMPATHY_INSTRUCTION}".strip()

    return system_instruction or EMPATHY_INSTRUCTION
