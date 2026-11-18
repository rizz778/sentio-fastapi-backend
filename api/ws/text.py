import asyncio
import json
import logging
from typing import Optional
from fastapi import APIRouter, Depends, Query, WebSocket

from api.deps import get_gemini_service
from api.ws.state import get_client_emotion
from prompts.chat import build_chat_system_instruction
from services.gemini_service import GeminiService

logger = logging.getLogger(__name__)

router = APIRouter()


def is_valid_turn(turn):
    return (
        isinstance(turn, dict)
        and "role" in turn
        and "parts" in turn
        and isinstance(turn["parts"], list)
        and all(isinstance(part, dict) and "text" in part for part in turn["parts"])
    )


def get_session_params(setup_data: dict, default_model: str):
    system_context = setup_data.get("systemContext") or {}
    custom_instruction = setup_data.get("systemInstruction")
    model = setup_data.get("model") or default_model or "gemini-3.5-flash-lite"
    if "live" in model.lower() or "audio" in model.lower():
        model = "gemini-3.5-flash-lite"

    final_instruction = build_chat_system_instruction(
        user_profile=system_context.get("user_profile"),
        conversation_history=system_context.get("conversation_history"),
        custom_instruction=custom_instruction,
    )
    return model, final_instruction


@router.websocket("/text")
async def text_websocket_endpoint(
    websocket: WebSocket,
    client_id: Optional[str] = Query(None),
    gemini_service: GeminiService = Depends(get_gemini_service),
):
    active_sessions = {}
    await websocket.accept()
    resolved_client_id = client_id or f"client_{id(websocket)}"

    try:
        try:
            setup_msg = await websocket.receive_text()
        except Exception as e:
            logger.warning(f"[TextWS] Client {resolved_client_id} disconnected before setup: {e}")
            return

        try:
            setup_data = json.loads(setup_msg)
        except json.JSONDecodeError:
            await websocket.send_json({"error": "Invalid JSON in setup message"})
            return

        if not isinstance(setup_data, dict):
            await websocket.send_json({"error": "Setup message must be a JSON object"})
            return

        if "setup" in setup_data:
            if setup_data["setup"] == {}:
                await websocket.send_json({"type": "setup_complete"})
            else:
                model, final_instruction = get_session_params(setup_data["setup"], gemini_service.model)
                active_sessions[resolved_client_id] = {
                    "model": model,
                    "system_instruction": final_instruction,
                }
                logger.info(f"[TextWS] Setup completed for {resolved_client_id} with model {model}")
                await websocket.send_json({"type": "setup_complete"})
        else:
            await websocket.send_json({"error": "First message must be setup"})
            return

        while True:
            msg = await websocket.receive_text()
            try:
                data = json.loads(msg)
            except json.JSONDecodeError:
                await websocket.send_json({"error": "Invalid JSON"})
                continue

            if not isinstance(data, dict):
                await websocket.send_json({"error": "Message must be a JSON object"})
                continue

            if "clientContent" in data:
                user_turns = data["clientContent"].get("turns", [])
                turn_complete = data["clientContent"].get("turnComplete", True)

                # Prioritize explicit emotion in payload, otherwise read from shared frame state
                latest_emotion = data["clientContent"].get("emotion") or get_client_emotion(resolved_client_id)
                emotion_context = None
                if latest_emotion:
                    emotion_context = {
                        "role": "user",
                        "parts": [{"text": f"My current emotion is {latest_emotion}."}],
                    }

                if emotion_context:
                    user_turns = user_turns + [emotion_context]

                if user_turns and all(is_valid_turn(turn) for turn in user_turns):
                    try:
                        session_info = active_sessions.get(resolved_client_id, {})
                        model_to_use = session_info.get("model", gemini_service.model)
                        system_instr = session_info.get("system_instruction")

                        logger.debug(f"[TextWS] Streaming response for {resolved_client_id} using {model_to_use}")
                        async for text_chunk in gemini_service.stream_text(
                            turns=user_turns,
                            system_instruction=system_instr,
                            model=model_to_use,
                        ):
                            await websocket.send_json({
                                "type": "text_chunk",
                                "content": text_chunk,
                                "complete": False,
                            })

                        # Mark text chunk complete
                        await websocket.send_json({
                            "type": "text_chunk",
                            "content": "",
                            "complete": True,
                        })
                        # Mark generation complete
                        await websocket.send_json({
                            "type": "generation_complete",
                            "message": "Response generation completed",
                        })

                    except Exception as e:
                        logger.error(f"[TextWS] Error streaming content for {resolved_client_id}: {e}")
                        await websocket.send_json({
                            "type": "error",
                            "message": f"Generation error: {str(e)}",
                        })
                else:
                    logger.error(f"[TextWS] Invalid user_turns structure: {user_turns}")
                    await websocket.send_json({"error": "Invalid message format for Gemini"})

            elif "setup" in data or data.get("type") == "setup":
                if "setup" in data and isinstance(data["setup"], dict):
                    model, final_instruction = get_session_params(data["setup"], gemini_service.model)
                    active_sessions[resolved_client_id] = {
                        "model": model,
                        "system_instruction": final_instruction,
                    }
                await websocket.send_json({"type": "setup_complete"})
            else:
                await websocket.send_json({"error": "Unsupported message format, expected 'clientContent'"})

    except Exception as e:
        logger.error(f"[TextWS] Error with client {resolved_client_id}: {e}")

    finally:
        if resolved_client_id in active_sessions:
            del active_sessions[resolved_client_id]
        logger.info(f"[TextWS] Cleanup completed for {resolved_client_id}")
