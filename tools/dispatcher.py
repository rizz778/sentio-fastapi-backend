"""
Tool Dispatcher
===============
Modular dispatcher for agent tools.
Takes tool name and arguments, locates the matched handler function via
the tool matcher registry, executes it, and returns the result safely.
"""

import logging
from typing import Any, Callable, Coroutine, Dict, Optional
from tools.matchers import TOOL_MATCHER, get_tool_handler

logger = logging.getLogger(__name__)


class ToolDispatcher:
    """
    Dispatcher responsible for validating and executing tools matched by name.
    """

    def __init__(self, matcher: Optional[Dict[str, Callable]] = None):
        self._matcher = matcher or TOOL_MATCHER.copy()

    def register(
        self,
        name: str,
        handler: Callable[..., Coroutine[Any, Any, Dict[str, Any]]],
    ) -> None:
        """Dynamically registers or overrides a tool handler."""
        self._matcher[name] = handler
        logger.debug(f"[ToolDispatcher] Registered tool handler: '{name}'")

    async def dispatch(
        self,
        tool_name: str,
        args: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Executes the matched function for the provided tool name and arguments.

        :param tool_name: The name of the tool invoked by the agent.
        :param args: Dictionary of arguments provided by the model.
        :param context: Execution context (user_id, session_id, auth_token, etc.).
        :return: Standardized result dictionary.
        """
        context = context or {}
        handler = self._matcher.get(tool_name) or get_tool_handler(tool_name)

        if not handler:
            logger.warning(f"[ToolDispatcher] No handler matched for tool: '{tool_name}'")
            return {
                "status": "error",
                "tool": tool_name,
                "error": f"Tool '{tool_name}' is not recognized or has no handler configured.",
            }

        logger.info(f"[ToolDispatcher] Executing tool '{tool_name}' with args={args}")
        try:
            result = await handler(args, context)
            return {
                "status": "success",
                "tool": tool_name,
                "result": result,
            }
        except Exception as e:
            logger.error(f"[ToolDispatcher] Error executing tool '{tool_name}': {e}", exc_info=True)
            return {
                "status": "error",
                "tool": tool_name,
                "error": str(e),
            }


tool_dispatcher = ToolDispatcher()
