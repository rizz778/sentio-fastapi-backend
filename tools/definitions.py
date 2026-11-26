"""
Tool Definitions
================
Declarations and Gemini FunctionDeclaration schemas for client and agent tools.
Kept strictly separate from handler implementation logic and dispatching.
"""

from typing import Any, Dict, List

RECALL_PAST_MEMORIES_DEF: Dict[str, Any] = {
    "name": "recall_past_memories",
    "description": (
        "Search and retrieve the user's past episodic memories (e.g. conflicts, achievements, "
        "past panic attacks, family/work stressors) to provide continuous, personalized therapeutic care."
    ),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "query": {
                "type": "STRING",
                "description": "Topic, person, or feeling to recall (e.g. 'conflict with manager', 'panic attack', 'mom').",
            },
            "limit": {
                "type": "INTEGER",
                "description": "Number of relevant memories to retrieve (default: 3).",
            },
        },
        "required": ["query"],
    },
}

UPDATE_MEMORY_RESOLUTION_DEF: Dict[str, Any] = {
    "name": "update_memory_resolution",
    "description": (
        "Update the status of an ongoing or unresolved past memory when the user reports "
        "that an issue was resolved, improved, or handled with a coping strategy."
    ),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "episode_id": {
                "type": "STRING",
                "description": "The unique ID of the episodic memory.",
            },
            "resolution_status": {
                "type": "STRING",
                "enum": ["unresolved", "ongoing", "resolved"],
                "description": "The new resolution status of the memory.",
            },
            "coping_attempted": {
                "type": "STRING",
                "description": "Any coping mechanism or action the user used to resolve or manage it.",
            },
        },
        "required": ["episode_id", "resolution_status"],
    },
}


def get_tool_declarations() -> List[Dict[str, Any]]:
    """Returns all available tool declarations for the Gemini Live agent."""
    return [
        RECALL_PAST_MEMORIES_DEF,
        UPDATE_MEMORY_RESOLUTION_DEF,
    ]
