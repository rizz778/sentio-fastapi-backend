import asyncio
import json
import logging
from typing import Any, Dict, List, Optional
from google import genai
from google.genai import types

from core.config import settings

logger = logging.getLogger(__name__)


class GeminiService:
    def __init__(self, api_key: str, model: Optional[str] = None):
        self.client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(api_version="v1alpha"),
        )
        self.model = model or settings.GEMINI_TEXT_MODEL

    async def stream_text(
        self,
        turns: List[Dict[str, Any]],
        system_instruction: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """Stream real-time text tokens using standard Gemini content generation."""
        contents = []
        for turn in turns:
            role = "user" if turn.get("role") == "user" else "model"
            parts = []
            for p in turn.get("parts", []):
                if isinstance(p, dict) and "text" in p and p["text"]:
                    parts.append(types.Part.from_text(text=p["text"]))
            if parts:
                contents.append(types.Content(role=role, parts=parts))

        model_to_use = model or self.model or "gemini-3.5-flash-lite"
        # If model is configured as an audio/live model, fallback to gemini-3.5-flash-lite for text
        if "live" in model_to_use.lower() or "audio" in model_to_use.lower():
            model_to_use = "gemini-3.5-flash-lite"

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.7,
        )

        try:
            response_stream = await self.client.aio.models.generate_content_stream(
                model=model_to_use,
                contents=contents,
                config=config,
            )
            async for chunk in response_stream:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            fallback_model = "gemini-3.1-flash-lite"
            if model_to_use != fallback_model:
                logger.warning(f"[GeminiService] Model {model_to_use} failed ({e}), falling back to {fallback_model}")
                response_stream = await self.client.aio.models.generate_content_stream(
                    model=fallback_model,
                    contents=contents,
                    config=config,
                )
                async for chunk in response_stream:
                    if chunk.text:
                        yield chunk.text
            else:
                raise e

    async def start_session(self, config: Any, model: Optional[str] = None):
        """Start a Gemini Live multimodal connection with sliding context window compression."""
        if isinstance(config, dict):
            config = types.LiveConnectConfig(
                **{k: v for k, v in config.items() if k != "context_window_compression"},
                context_window_compression=types.ContextWindowCompressionConfig(
                    sliding_window=types.SlidingWindow()
                ),
            )
        else:
            config.context_window_compression = types.ContextWindowCompressionConfig(
                sliding_window=types.SlidingWindow()
            )

        model_to_use = model or self.model
        session_cm = self.client.aio.live.connect(model=model_to_use, config=config)
        session = await session_cm.__aenter__()
        return session, session_cm

    async def send_client_content(self, session: Any, turns: List[Dict[str, Any]], turn_complete: bool = True):
        """Validate structure and send user content turns to Gemini session."""
        if not isinstance(turns, list):
            logger.error(f"[GeminiService] 'turns' is not a list: {turns}")
            return

        for i, turn in enumerate(turns):
            if not isinstance(turn, dict) or "role" not in turn or "parts" not in turn:
                logger.error(f"[GeminiService] Invalid turn at index {i}: {turn}")
                return
            if not isinstance(turn["parts"], list):
                logger.error(f"[GeminiService] 'parts' is not a list at index {i}: {turn['parts']}")
                return
            for j, part in enumerate(turn["parts"]):
                if not isinstance(part, dict) or "text" not in part:
                    logger.error(f"[GeminiService] Invalid part at turn {i}, part {j}: {part}")
                    return

        logger.debug(f"[GeminiService] Sending turns to Gemini (turn_complete={turn_complete}): {turns}")
        await session.send_client_content(turns=turns, turn_complete=turn_complete)

    async def _safe_websocket_send(self, websocket: Any, client_id: str, data_dict: Dict[str, Any]) -> bool:
        """Safely send data to websocket with error handling."""
        try:
            if not isinstance(data_dict, dict):
                logger.error(f"[Gemini] Expected dict for websocket send, got {type(data_dict)} for {client_id}")
                return False

            if not websocket or not hasattr(websocket, "send_json"):
                logger.error(f"[Gemini] Invalid websocket object for {client_id}")
                return False

            await websocket.send_json(data_dict)
            return True
        except Exception as e:
            logger.error(f"[Gemini] Safe websocket send failed for {client_id}: {e}")
            return False

    async def receive_responses(
        self,
        session: Any,
        websocket: Any,
        client_id: str,
        active_sessions: Dict[str, Any],
    ):
        """Stream responses from Gemini Live API and forward to the client WebSocket."""
        if not websocket:
            logger.error(f"[Gemini] Invalid websocket object for client {client_id}")
            return

        if not isinstance(client_id, str):
            logger.error(f"[Gemini] Invalid client_id type: {type(client_id)}")
            return

        if not isinstance(active_sessions, dict):
            logger.error(f"[Gemini] Invalid active_sessions type: {type(active_sessions)}")
            return

        try:
            if client_id not in active_sessions or not isinstance(active_sessions[client_id], dict):
                active_sessions[client_id] = {}

            while True:
                async for response in session.receive():
                    if isinstance(response, bytes):
                        response_str = response.decode("utf-8", errors="replace")
                        logger.error(f"[Gemini] Unexpected bytes response for {client_id}: {response_str}")
                        await self._safe_websocket_send(websocket, client_id, {
                            "type": "error",
                            "message": f"Unexpected Gemini API response: {response_str}",
                        })
                        continue

                    elif isinstance(response, str):
                        logger.error(f"[Gemini] Unexpected string response for {client_id}: {response}")
                        await self._safe_websocket_send(websocket, client_id, {
                            "type": "error",
                            "message": f"Unexpected Gemini API response: {response}",
                        })
                        continue

                    elif hasattr(response, "session_resumption_update") and response.session_resumption_update:
                        handle = response.session_resumption_update.new_handle
                        resumable = response.session_resumption_update.resumable

                        if not isinstance(active_sessions.get(client_id), dict):
                            active_sessions[client_id] = {}

                        active_sessions[client_id]["session_handle"] = handle
                        await self._safe_websocket_send(websocket, client_id, {
                            "type": "session_resumption_update",
                            "handle": handle,
                            "resumable": resumable,
                        })

                    elif getattr(response, "go_away", None):
                        logger.info(f"[GeminiService] Received go_away for {client_id}: {response.go_away}")
                        session_handle = None
                        client_data = active_sessions.get(client_id)
                        if isinstance(client_data, dict):
                            session_handle = client_data.get("session_handle")

                        await self._safe_websocket_send(websocket, client_id, {
                            "type": "session_warning",
                            "message": f"Session ending in: {getattr(response.go_away, 'time_left', None)}",
                            "go_away": str(response.go_away),
                            "resume_handle": session_handle,
                        })

                    elif getattr(response, "text", None) is not None:
                        logger.debug(f"[GeminiService] Received text for {client_id}: {response.text}")
                        await self._safe_websocket_send(websocket, client_id, {
                            "type": "text_chunk",
                            "content": response.text,
                            "complete": False,
                        })

                        if getattr(response, "generation_complete", False) or (
                            hasattr(response, "server_content")
                            and getattr(response.server_content, "generation_complete", False)
                        ):
                            await self._safe_websocket_send(websocket, client_id, {
                                "type": "text_chunk",
                                "content": "",
                                "complete": True,
                            })

                    elif getattr(response, "server_content", None):
                        if response.server_content.generation_complete:
                            await self._safe_websocket_send(websocket, client_id, {
                                "type": "generation_complete",
                                "message": "Response generation completed",
                            })
                        if response.server_content.interrupted:
                            await self._safe_websocket_send(websocket, client_id, {
                                "type": "interrupted",
                                "message": "Generation was interrupted",
                            })

                    elif hasattr(response, "tool_call") and response.tool_call:
                        await self._safe_websocket_send(websocket, client_id, {
                            "type": "tool_call",
                            "tool_call": str(response.tool_call),
                        })

                    elif getattr(response, "usage_metadata", None):
                        usage = response.usage_metadata
                        details = []
                        for detail in getattr(usage, "response_tokens_details", []):
                            modality = getattr(detail, "modality", "unknown")
                            if hasattr(modality, "value"):
                                modality = modality.value
                            elif hasattr(modality, "name"):
                                modality = modality.name
                            else:
                                modality = str(modality)

                            details.append({
                                "modality": modality,
                                "token_count": getattr(detail, "token_count", 0),
                            })

                        await self._safe_websocket_send(websocket, client_id, {
                            "type": "usage_update",
                            "total_tokens": usage.total_token_count,
                            "details": details,
                        })
        except Exception as e:
            logger.error(f"[Gemini] Error listening to Live API for {client_id}: {e}")
            await self._safe_websocket_send(websocket, client_id, {
                "type": "error",
                "message": str(e),
            })
