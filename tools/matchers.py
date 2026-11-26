"""
Tool Matchers & Handlers
========================
Contains the actual function implementations for agent tools and the matcher
mapping tool names to their corresponding executable functions.
"""

import logging
from typing import Any, Callable, Coroutine, Dict, Optional
from clients.node.memory import memory_client

logger = logging.getLogger(__name__)


async def handle_recall_past_memories(
    args: Dict[str, Any],
    context: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Retrieves relevant past episodic memories using vector search.
    Invokes memory_service to compute query embeddings.
    """
    user_id = context.get("user_id")
    auth_token = context.get("auth_token")
    query = (args.get("query") or "").strip()
    limit = int(args.get("limit", 3))

    if not user_id:
        return {"error": "User identity not found in execution context"}
    if not query:
        return {"error": "Search query is required"}

    try:
        from services.memory_service import memory_service

        matches = await memory_service.query_episodes(
            user_id=user_id,
            query_text=query,
            limit=limit,
            auth_token=auth_token,
        )

        if not matches:
            return {
                "count": 0,
                "message": f"No past memories found matching topic '{query}'.",
                "memories": [],
            }

        formatted_memories = []
        for ep in matches:
            formatted_memories.append({
                "id": str(ep.get("_id", "")),
                "summary": ep.get("eventSummary", ""),
                "eventType": ep.get("eventType", "general"),
                "emotions": ep.get("dominantEmotions", []),
                "peakDistress": ep.get("peakDistressScore", 0),
                "coping": ep.get("copingAttempted", ""),
                "status": ep.get("resolutionStatus", "unresolved"),
                "people": ep.get("peopleMentioned", []),
                "timestamp": ep.get("timestamp"),
            })

        return {
            "count": len(formatted_memories),
            "memories": formatted_memories,
        }
    except Exception as e:
        logger.error(f"[ToolMatcher] recall_past_memories failed: {e}", exc_info=True)
        return {"error": f"Failed to retrieve past memories: {str(e)}"}


async def handle_update_memory_resolution(
    args: Dict[str, Any],
    context: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Updates an episodic memory's resolution status and coping strategy.
    """
    user_id = context.get("user_id")
    auth_token = context.get("auth_token")
    episode_id = (args.get("episode_id") or "").strip()
    resolution_status = (args.get("resolution_status") or "resolved").strip()
    coping_attempted = args.get("coping_attempted")

    if not user_id:
        return {"error": "User identity not found in execution context"}
    if not episode_id:
        return {"error": "episode_id is required"}

    try:
        updated = await memory_client.patch_episode(
            user_id=user_id,
            episode_id=episode_id,
            resolution_status=resolution_status,
            coping_attempted=coping_attempted,
            auth_token=auth_token,
        )

        if updated:
            return {
                "success": True,
                "message": f"Episode '{episode_id}' status marked as {resolution_status}.",
                "episode": updated,
            }
        return {"error": f"Could not find or update episode '{episode_id}'"}
    except Exception as e:
        logger.error(f"[ToolMatcher] update_memory_resolution failed: {e}", exc_info=True)
        return {"error": f"Failed to update memory resolution: {str(e)}"}


# Tool matcher: maps tool name to actual executable function
TOOL_MATCHER: Dict[str, Callable[[Dict[str, Any], Dict[str, Any]], Coroutine[Any, Any, Dict[str, Any]]]] = {
    "recall_past_memories": handle_recall_past_memories,
    "update_memory_resolution": handle_update_memory_resolution,
}


def get_tool_handler(tool_name: str) -> Optional[Callable]:
    """Returns the matching execution function for a given tool name."""
    return TOOL_MATCHER.get(tool_name)
