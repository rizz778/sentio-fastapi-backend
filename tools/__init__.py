"""
Tools Package
=============
Modular tool definitions, matchers, and dispatcher for Gemini Live agent.
"""

from tools.definitions import (
    RECALL_PAST_MEMORIES_DEF,
    UPDATE_MEMORY_RESOLUTION_DEF,
    get_tool_declarations,
)
from tools.matchers import (
    TOOL_MATCHER,
    get_tool_handler,
    handle_recall_past_memories,
    handle_update_memory_resolution,
)
from tools.dispatcher import ToolDispatcher, tool_dispatcher

__all__ = [
    "RECALL_PAST_MEMORIES_DEF",
    "UPDATE_MEMORY_RESOLUTION_DEF",
    "get_tool_declarations",
    "TOOL_MATCHER",
    "get_tool_handler",
    "handle_recall_past_memories",
    "handle_update_memory_resolution",
    "ToolDispatcher",
    "tool_dispatcher",
]
