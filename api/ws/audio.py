import json
import logging
from fastapi import APIRouter, WebSocket
from services.audio_service import AudioConnectionManager

from prompts.audio import build_audio_system_instruction

logger = logging.getLogger(__name__)

router = APIRouter()

audio_manager = AudioConnectionManager()
audio_client_emotions = {}  # Store latest emotion per audio client


@router.websocket("/audio")
async def audio_websocket_endpoint(websocket: WebSocket):
    client_id = f"audio_client_{id(websocket)}"
    try:
        await audio_manager.connect(websocket)
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            if message.get("type") == "setup":
                config = message.get("config", {})
                system_context = config.get("systemContext") or {}
                user_profile = system_context.get("user_profile")
                conversation_history = system_context.get("conversation_history")

                system_instruction = build_audio_system_instruction(
                    user_profile=user_profile,
                    conversation_history=conversation_history,
                )

                # Start Gemini session
                gemini_client = audio_manager.active_connections.get(websocket)
                if gemini_client:
                    await gemini_client.send_setup(config, system_instruction)
                await websocket.send_json({"type": "setup_complete"})

            elif message.get("type") == "audio_input":
                audio_data = message.get("audio_data", "")
                emotion = message.get("emotion")
                if emotion:
                    audio_client_emotions[client_id] = emotion
                end_of_stream = message.get("end_of_stream", True)
                if audio_data:
                    gemini_client = audio_manager.active_connections.get(websocket)
                    if gemini_client:
                        await gemini_client.send_audio(audio_data, end_of_stream=end_of_stream)

            elif message.get("type") == "text_input":
                text = message.get("text", "")
                latest_emotion = audio_client_emotions.get(client_id)
                if latest_emotion:
                    emotion_context = f"My current emotion is {latest_emotion}. "
                    text = f"{emotion_context}{text}"
                if text:
                    gemini_client = audio_manager.active_connections.get(websocket)
                    if gemini_client:
                        await gemini_client.send_text(text)

    except Exception as e:
        logger.error(f"[AudioWebSocket] Connection error for {client_id}: {e}", exc_info=True)
        await audio_manager.disconnect(websocket)
    finally:
        if client_id in audio_client_emotions:
            del audio_client_emotions[client_id]
