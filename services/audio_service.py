import asyncio
import json
import logging
from typing import Any, Dict
from fastapi import WebSocket
import websockets

from core.config import settings

logger = logging.getLogger(__name__)


class GeminiAudioService:
    def __init__(self):
        self.api_key = settings.GEMINI_API_KEY
        self.websocket = None
        self.connected = False
        self.config = None
        self.system_prompt = None

    async def connect(self):
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is not configured")
        url = f"wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1alpha.GenerativeService.BidiGenerateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        logger.info("[GeminiAudioService] Connecting to Gemini WebSocket...")
        self.websocket = await websockets.connect(url, additional_headers=headers)
        self.connected = True
        logger.info("[GeminiAudioService] Connected to Gemini WebSocket.")

    async def send_setup(self, config: Dict[str, Any], system_prompt: str):
        self.config = config
        self.system_prompt = system_prompt
        model_name = config.get("model") or settings.GEMINI_AUDIO_MODEL
        if not model_name.startswith("models/"):
            model_name = f"models/{model_name}"

        setup_message = {
            "setup": {
                "model": model_name,
                "generationConfig": {
                    "responseModalities": config.get("response_modalities", ["AUDIO"]),
                    "speechConfig": {
                        "voiceConfig": {
                            "prebuiltVoiceConfig": {
                                "voiceName": config.get("voice_name", "Puck")
                            }
                        }
                    },
                    "enable_affective_dialog": True,
                },
                "inputAudioTranscription": {},
                "outputAudioTranscription": {},
                "systemInstruction": {
                    "parts": [{"text": system_prompt}]
                },
                "tools": [],
            }
        }
        await self.websocket.send(json.dumps(setup_message))
        logger.info("[GeminiAudioService] Setup message sent.")

    async def ensure_connected(self):
        if not self.connected or not self.websocket:
            logger.info("[GeminiAudioService] Connection not active, reconnecting...")
            await self.connect()
            if self.config and self.system_prompt:
                await self.send_setup(self.config, self.system_prompt)

    async def send_audio(self, audio_data: str, end_of_stream: bool = False):
        """Send audio data to Gemini using realtime input. Optionally signal end of stream."""
        await self.ensure_connected()
        logger.debug(f"[GeminiAudioService] Sending audio data, size: {len(audio_data)} chars")
        message = {
            "realtimeInput": {
                "mediaChunks": [
                    {
                        "mimeType": "audio/pcm;rate=16000",
                        "data": audio_data,
                    }
                ]
            }
        }
        await self.websocket.send(json.dumps(message))
        logger.debug("[GeminiAudioService] Audio data sent successfully.")

    async def send_text(self, text: str):
        await self.ensure_connected()
        logger.info(f"[GeminiAudioService] Sending text: {text}")
        message = {
            "clientContent": {
                "turns": [
                    {
                        "role": "user",
                        "parts": [{"text": text}],
                    }
                ],
                "turnComplete": True,
            }
        }
        await self.websocket.send(json.dumps(message))
        logger.info("[GeminiAudioService] Text message sent.")

    async def receive_message(self):
        if not self.connected or not self.websocket:
            logger.warning("[GeminiAudioService] Tried to receive message but not connected.")
            return None
        try:
            message = await self.websocket.recv()
            logger.debug(f"[GeminiAudioService] Received message: {message[:120]}...")
            return json.loads(message)
        except Exception as e:
            logger.error(f"[GeminiAudioService] Error receiving message: {e}")
            self.connected = False
            return None

    async def close(self):
        if self.websocket:
            logger.info("[GeminiAudioService] Closing Gemini WebSocket.")
            await self.websocket.close()
        self.connected = False


class AudioConnectionManager:
    def __init__(self):
        self.active_connections: Dict[WebSocket, GeminiAudioService] = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        logger.info(f"[AudioConnectionManager] New client connected: {id(websocket)}")
        gemini_client = GeminiAudioService()
        await gemini_client.connect()
        self.active_connections[websocket] = gemini_client
        asyncio.create_task(self.listen_to_gemini(websocket, gemini_client))

    async def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            logger.info(f"[AudioConnectionManager] Disconnecting client: {id(websocket)}")
            gemini_client = self.active_connections[websocket]
            await gemini_client.close()
            del self.active_connections[websocket]

    async def send_message(self, websocket: WebSocket, message: dict):
        try:
            logger.debug(f"[AudioConnectionManager] Sending message to client: {message.get('type')}")
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"[AudioConnectionManager] Error sending message to client: {e}")

    async def listen_to_gemini(self, websocket: WebSocket, gemini_client: GeminiAudioService):
        logger.info(f"[AudioConnectionManager] Listening to Gemini for client: {id(websocket)}")
        while gemini_client.connected:
            try:
                message = await gemini_client.receive_message()
                if message is None:
                    logger.info(f"[AudioConnectionManager] Gemini connection closed for client: {id(websocket)}")
                    break
                await self.process_gemini_message(websocket, message)
            except Exception as e:
                logger.error(f"[AudioConnectionManager] Error in Gemini listener: {e}")
                break

    async def process_gemini_message(self, websocket: WebSocket, message: dict):
        try:
            logger.debug(f"[AudioConnectionManager] Processing Gemini message: {str(message)[:100]}...")
            if "setupComplete" in message:
                await self.send_message(websocket, {"type": "info", "message": "Gemini setup complete"})
            elif "serverContent" in message:
                content = message["serverContent"]
                if "modelTurn" in content:
                    parts = content["modelTurn"].get("parts", [])
                    for part in parts:
                        if "inlineData" in part:
                            inline_data = part["inlineData"]
                            mime_type = inline_data.get("mimeType", "audio/pcm")
                            await self.send_message(websocket, {
                                "type": "audio",
                                "audio_data": inline_data.get("data", ""),
                                "mime_type": mime_type,
                            })
                        elif "text" in part:
                            await self.send_message(websocket, {
                                "type": "transcript",
                                "text": part["text"],
                            })
                if "outputTranscription" in content:
                    await self.send_message(websocket, {
                        "type": "output_transcription",
                        "text": content["outputTranscription"].get("text", ""),
                    })
                if "inputTranscription" in content:
                    await self.send_message(websocket, {
                        "type": "input_transcription",
                        "text": content["inputTranscription"].get("text", ""),
                    })
                if content.get("generationComplete"):
                    await self.send_message(websocket, {"type": "info", "message": "generationComplete"})
                if content.get("turnComplete"):
                    await self.send_message(websocket, {"type": "turnComplete"})
            elif "toolCall" in message:
                tool_call = message["toolCall"]
                await self.send_message(websocket, {"type": "info", "message": f"Tool call: {tool_call}"})
            elif "interrupted" in message:
                await self.send_message(websocket, {"type": "info", "message": "Conversation interrupted"})
            elif "realtimeInput" in message:
                realtime_input = message["realtimeInput"]
                if "userTranscript" in realtime_input:
                    await self.send_message(websocket, {"type": "user_transcript", "text": realtime_input["userTranscript"]})
            elif "error" in message:
                error = message["error"]
                await self.send_message(websocket, {"type": "error", "message": f"Gemini error: {error.get('message', 'Unknown error')}"})
            elif "goAway" in message:
                go_away = message["goAway"]
                await self.send_message(websocket, {"type": "info", "message": f"Session ending in {go_away.get('timeLeft', 'unknown')} seconds"})
            else:
                logger.debug(f"[AudioConnectionManager] Unhandled Gemini message type: {message}")
        except Exception as e:
            logger.error(f"[AudioConnectionManager] Error processing message: {e}")
            await self.send_message(websocket, {"type": "error", "message": f"Error processing response: {str(e)}"})
