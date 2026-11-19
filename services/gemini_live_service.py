"""
Unified Gemini Live Service
============================
Single persistent Gemini Live API session per user that handles:
- Audio in/out (realtime PCM streaming)
- Text in/out (typed messages via client_content)
- Real-time transcriptions (user & model)
- Barge-in / interruption
- Session resumption on reconnect
- Sliding window context compression

Replaces the old separate audio_service.py and gemini_service.py.
"""

import asyncio
import base64
import logging
import time
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional

from google import genai
from google.genai import types

from core.config import settings
from prompts.chat import build_chat_system_instruction
from services.redis_service import redis_service

logger = logging.getLogger(__name__)


class GeminiLiveSession:
    """
    Wraps one persistent Gemini Live API connection for a single user session.
    Produces outbound events (audio, transcript, interruption) via an async callback.
    """

    def __init__(
        self,
        session_id: str,
        user_id: str,
        on_event: Callable[[Dict[str, Any]], Any],
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.session_id = session_id
        self.user_id = user_id
        self.on_event = on_event
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.model = model or settings.GEMINI_AUDIO_MODEL

        self._client: Optional[genai.Client] = None
        self._session: Any = None
        self._session_cm: Any = None
        self._listener_task: Optional[asyncio.Task] = None
        self._closed = False

    async def connect(
        self,
        system_instruction: Optional[str] = None,
        voice_name: Optional[str] = None,
        response_modalities: Optional[List[str]] = None,
        resume_handle: Optional[str] = None,
    ) -> None:
        """
        Opens the Gemini Live session with the client-specified modalities (default AUDIO).
        Voice name and base system instruction fall back to FastAPI backend settings.
        Fetches updated conversation history directly from Redis.
        """
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is not configured")

        self._client = genai.Client(
            api_key=self.api_key,
            http_options=types.HttpOptions(api_version="v1alpha"),
        )

        modalities = [m.upper() for m in response_modalities] if response_modalities else ["AUDIO"]
        selected_voice = voice_name or getattr(settings, "GEMINI_VOICE_NAME", "Puck")
        effective_instruction = system_instruction or getattr(settings, "DEFAULT_SYSTEM_INSTRUCTION", "You are a Therapist")

        speech_config = None
        if "AUDIO" in modalities:
            speech_config = types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=selected_voice
                    )
                )
            )

        # Build session config
        config = types.LiveConnectConfig(
            response_modalities=modalities,
            speech_config=speech_config,
            system_instruction=types.Content(
                parts=[types.Part.from_text(text=effective_instruction)]
            ),
            enable_affective_dialog=True,
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
            context_window_compression=types.ContextWindowCompressionConfig(
                sliding_window=types.SlidingWindow()
            ),
        )

        # Attach session resumption handle if rehydrating
        if resume_handle:
            config.session_resumption = types.SessionResumptionConfig(
                handle=resume_handle
            )

        logger.info(
            f"[GeminiLive] Connecting session {self.session_id} with model={self.model}, "
            f"modalities={modalities}, voice={voice_name}"
        )

        self._session_cm = self._client.aio.live.connect(
            model=self.model, config=config
        )
        self._session = await self._session_cm.__aenter__()

        # Fetch and inject updated conversation history directly from Redis
        await self.inject_history_from_redis()

        # Start the background listener that reads all Gemini responses
        self._listener_task = asyncio.create_task(self._listen_loop())

        logger.info(
            f"[GeminiLive] Session {self.session_id} connected and listener started."
        )

    async def inject_history_from_redis(self) -> None:
        """Fetches the latest sliding window from Redis and injects it as context turns into the session."""
        if not self._session or self._closed:
            return

        messages = await redis_service.get_sliding_window(self.session_id)
        if not messages:
            return

        turns = []
        for msg in messages:
            role = "user" if msg.get("sender") == "user" else "model"
            text = msg.get("text", "")
            if text:
                turns.append(
                    types.Content(
                        role=role,
                        parts=[types.Part.from_text(text=text)],
                    )
                )

        if turns:
            try:
                await self._session.send_client_content(
                    turns=turns, turn_complete=True
                )
                logger.info(
                    f"[GeminiLive] Injected {len(turns)} history turns from Redis for session {self.session_id}"
                )
            except Exception as e:
                logger.warning(
                    f"[GeminiLive] Failed to inject history from Redis for {self.session_id}: {e}"
                )

    # ── Input Methods ─────────────────────────────────────────────────────────

    async def send_audio(self, pcm_bytes: bytes, mime_type: str = "audio/pcm;rate=16000") -> None:
        """Stream raw PCM audio bytes into the live session."""
        if self._closed or not self._session:
            return
        try:
            await self._session.send_realtime_input(
                audio=types.Blob(data=pcm_bytes, mime_type=mime_type)
            )
        except Exception as e:
            logger.error(f"[GeminiLive] Audio send failed for {self.session_id}: {e}")

    async def send_text(self, text: str) -> None:
        """Send a typed text turn into the live session."""
        if self._closed or not self._session:
            return
        try:
            turns = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=text)],
                )
            ]
            await self._session.send_client_content(
                turns=turns, turn_complete=True
            )
            logger.debug(f"[GeminiLive] Text sent for {self.session_id}: {text[:80]}")
        except Exception as e:
            logger.error(f"[GeminiLive] Text send failed for {self.session_id}: {e}")

    async def inject_emotion_context(self, emotion: str) -> None:
        """Silently injects an emotion shift cue into the conversation context."""
        if self._closed or not self._session:
            return
        context_text = (
            f"[System Context: User's facial expression now shows '{emotion}'. "
            f"Adapt your tone and response with appropriate empathy.]"
        )
        try:
            turns = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=context_text)],
                )
            ]
            await self._session.send_client_content(
                turns=turns, turn_complete=False
            )
        except Exception as e:
            logger.warning(f"[GeminiLive] Emotion context injection failed: {e}")

    # ── Response Listener ─────────────────────────────────────────────────────

    async def _listen_loop(self) -> None:
        """
        Background task that continuously reads Gemini Live responses
        and dispatches them as structured events via self.on_event().
        """
        try:
            while not self._closed and self._session:
                async for response in self._session.receive():
                    if self._closed:
                        break

                    # Session resumption handle update
                    if (
                        hasattr(response, "session_resumption_update")
                        and response.session_resumption_update
                    ):
                        handle = response.session_resumption_update.new_handle
                        await self.on_event({
                            "type": "resume_handle",
                            "handle": handle,
                            "resumable": response.session_resumption_update.resumable,
                        })
                        continue

                    # Go-away (session timeout warning)
                    if getattr(response, "go_away", None):
                        await self.on_event({
                            "type": "go_away",
                            "time_left": str(
                                getattr(response.go_away, "time_left", "unknown")
                            ),
                        })
                        continue

                    # Server content (audio, transcripts, generation status)
                    server_content = getattr(response, "server_content", None)
                    if server_content:
                        # Model turn parts (audio and/or text)
                        model_turn = getattr(server_content, "model_turn", None)
                        if model_turn and model_turn.parts:
                            for part in model_turn.parts:
                                # Audio output
                                if part.inline_data:
                                    audio_bytes = part.inline_data.data
                                    mime = part.inline_data.mime_type or "audio/pcm;rate=24000"
                                    await self.on_event({
                                        "type": "audio_chunk",
                                        "data": audio_bytes,
                                        "mime_type": mime,
                                    })
                                # Text output
                                if part.text:
                                    await self.on_event({
                                        "type": "model_transcript",
                                        "text": part.text,
                                        "is_final": False,
                                    })

                        # Output transcription (model speech-to-text)
                        output_transcription = getattr(
                            server_content, "output_transcription", None
                        )
                        if output_transcription and output_transcription.text:
                            await self.on_event({
                                "type": "model_transcript",
                                "text": output_transcription.text,
                                "is_final": True,
                            })

                        # Input transcription (user speech-to-text)
                        input_transcription = getattr(
                            server_content, "input_transcription", None
                        )
                        if input_transcription and input_transcription.text:
                            await self.on_event({
                                "type": "user_transcript",
                                "text": input_transcription.text,
                                "is_final": True,
                            })

                        # Interruption / barge-in
                        if getattr(server_content, "interrupted", False):
                            await self.on_event({
                                "type": "interrupted",
                                "timestamp_ms": int(time.time() * 1000),
                            })

                        # Generation complete
                        if getattr(server_content, "generation_complete", False):
                            await self.on_event({
                                "type": "generation_complete",
                            })

                        # Turn complete
                        if getattr(server_content, "turn_complete", False):
                            await self.on_event({
                                "type": "turn_complete",
                            })

                        continue

                    # Usage metadata
                    if getattr(response, "usage_metadata", None):
                        usage = response.usage_metadata
                        await self.on_event({
                            "type": "usage_update",
                            "total_tokens": getattr(usage, "total_token_count", 0),
                        })
                        continue

        except asyncio.CancelledError:
            logger.info(f"[GeminiLive] Listener cancelled for session {self.session_id}")
        except Exception as e:
            if not self._closed:
                logger.error(
                    f"[GeminiLive] Listener error for session {self.session_id}: {e}",
                    exc_info=True,
                )
                await self.on_event({
                    "type": "error",
                    "message": str(e),
                })

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def close(self) -> None:
        """Cleanly shuts down the Gemini Live session and listener."""
        if self._closed:
            return
        self._closed = True

        if self._listener_task and not self._listener_task.done():
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass

        if self._session_cm:
            try:
                await self._session_cm.__aexit__(None, None, None)
            except Exception as e:
                logger.warning(
                    f"[GeminiLive] Error closing session {self.session_id}: {e}"
                )

        self._session = None
        self._session_cm = None
        logger.info(f"[GeminiLive] Session {self.session_id} closed.")
