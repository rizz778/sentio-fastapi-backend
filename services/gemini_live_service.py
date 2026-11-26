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
from tools.definitions import get_tool_declarations
from tools.dispatcher import tool_dispatcher

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
        auth_token: Optional[str] = None,
    ):
        self.session_id = session_id
        self.user_id = user_id
        self.on_event = on_event
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.model = model
        self.auth_token = auth_token

        self._client: Optional[genai.Client] = None
        self._session: Any = None
        self._session_cm: Any = None
        self._listener_task: Optional[asyncio.Task] = None
        self._keepalive_task: Optional[asyncio.Task] = None
        self._closed = False
        self._system_instruction: Optional[str] = None
        self._voice_name: Optional[str] = None
        self._response_modalities: Optional[List[str]] = None
        self._resume_handle: Optional[str] = None
        self._last_activity_time: float = time.time()

    @property
    def is_connected(self) -> bool:
        return (
            self._session is not None
            and not self._closed
            and self._listener_task is not None
            and not self._listener_task.done()
        )

    async def ensure_connected(self) -> None:
        """Auto-reconnects if the session dropped or went idle due to inactivity."""
        if self.is_connected or self._closed:
            return

        logger.info(f"[GeminiLive] Auto-reconnecting idle/dormant session {self.session_id}...")
        if self._listener_task and not self._listener_task.done():
            self._listener_task.cancel()
        if self._keepalive_task and not self._keepalive_task.done():
            self._keepalive_task.cancel()

        cached_handle = await redis_service.get_gemini_resume_handle(self.session_id)
        resume_h = cached_handle or self._resume_handle
        try:
            await self.connect(
                system_instruction=self._system_instruction,
                voice_name=self._voice_name,
                response_modalities=self._response_modalities,
                resume_handle=resume_h,
            )
        except Exception as e:
            err_str = str(e).lower()
            if resume_h and ("not found" in err_str or "expired" in err_str or "1008" in err_str):
                logger.warning(f"[GeminiLive] Resume handle expired ({e}), connecting fresh...")
                await redis_service.clear_gemini_resume_handle(self.session_id)
                self._resume_handle = None
                await self.connect(
                    system_instruction=self._system_instruction,
                    voice_name=self._voice_name,
                    response_modalities=self._response_modalities,
                    resume_handle=None,
                )
            else:
                logger.error(f"[GeminiLive] Auto-reconnect failed for {self.session_id}: {e}")
                raise e

    async def _keepalive_loop(self) -> None:
        """
        Periodically sends a tiny frame of digital silence (PCM) if no input has been
        sent in the last 20 seconds, preventing Gemini Live's 60-second inactivity disconnect.
        """
        silence_chunk = b"\x00" * 320  # 10ms of 16kHz 16-bit mono PCM silence
        try:
            while not self._closed:
                await asyncio.sleep(15)
                if self._closed or not self.is_connected:
                    continue
                if time.time() - self._last_activity_time >= 20:
                    try:
                        await self._session.send_realtime_input(
                            audio=types.Blob(data=silence_chunk, mime_type="audio/pcm;rate=16000")
                        )
                        self._last_activity_time = time.time()
                        logger.debug(f"[GeminiLive] Keepalive ping sent for {self.session_id}")
                    except Exception as e:
                        logger.debug(f"[GeminiLive] Keepalive ping error: {e}")
        except asyncio.CancelledError:
            pass

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
            http_options=types.HttpOptions(api_version=getattr(settings, "GEMINI_API_VERSION", "v1beta")),
        )

        modalities = [m.upper() for m in response_modalities] if response_modalities else ["AUDIO"]
        selected_voice = voice_name or getattr(settings, "GEMINI_VOICE_NAME", "Puck")
        effective_instruction = system_instruction or getattr(settings, "DEFAULT_SYSTEM_INSTRUCTION", "You are a Therapist")

        # Gemini Live conversational model requires response_modalities=["AUDIO"] on bidi streaming.
        # It generates both high-quality audio streaming and real-time speech-to-text transcripts simultaneously.
        live_modalities = ["AUDIO"]
        effective_model = self.model or getattr(settings, "GEMINI_AUDIO_MODEL", "gemini-3.8-live")

        speech_config = types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name=selected_voice
                )
            )
        )

        # Build session config
        config_kwargs = {
            "response_modalities": live_modalities,
            "speech_config": speech_config,
            "input_audio_transcription": types.AudioTranscriptionConfig(),
            "output_audio_transcription": types.AudioTranscriptionConfig(),
            "system_instruction": types.Content(
                parts=[types.Part.from_text(text=effective_instruction)]
            ),
            "context_window_compression": types.ContextWindowCompressionConfig(
                sliding_window=types.SlidingWindow()
            ),
        }

        # Register therapeutic agent tools (memory retrieval & status updates)
        tool_declarations = get_tool_declarations()
        if tool_declarations:
            config_kwargs["tools"] = [types.Tool(function_declarations=tool_declarations)]

        # Attach session resumption handle if rehydrating
        if resume_handle:
            config_kwargs["session_resumption"] = types.SessionResumptionConfig(
                handle=resume_handle
            )

        config = types.LiveConnectConfig(**config_kwargs)

        logger.info(
            f"[GeminiLive] Connecting session {self.session_id} with model={effective_model}, "
            f"modalities={live_modalities}, voice={selected_voice}"
        )

        self._session_cm = self._client.aio.live.connect(
            model=effective_model, config=config
        )
        self._session = await self._session_cm.__aenter__()

        self._system_instruction = effective_instruction
        self._voice_name = selected_voice
        self._response_modalities = live_modalities
        if resume_handle:
            self._resume_handle = resume_handle
        self._last_activity_time = time.time()

        # History and user profile are already injected into system_instruction at connection time.
        # Starting the background listener that reads all Gemini Live responses.
        self._listener_task = asyncio.create_task(self._listen_loop())
        if not self._keepalive_task or self._keepalive_task.done():
            self._keepalive_task = asyncio.create_task(self._keepalive_loop())

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
        if self._closed:
            return
        await self.ensure_connected()
        if not self._session:
            return
        self._last_activity_time = time.time()
        try:
            await self._session.send_realtime_input(
                audio=types.Blob(data=pcm_bytes, mime_type=mime_type)
            )
        except Exception as e:
            logger.error(f"[GeminiLive] Audio send failed for {self.session_id}: {e}")

    async def send_text(self, text: str) -> None:
        """Send a typed text turn into the live session."""
        if self._closed:
            return
        await self.ensure_connected()
        if not self._session:
            return
        self._last_activity_time = time.time()
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

    
    async def kickoff_greeting(
        self,
        unresolved_episodes: Optional[List[Dict[str, Any]]] = None,
        user_name: Optional[str] = None,
    ) -> None:
        """
        Sends an initial directive so Gemini Live proactively speaks first with an empathetic
        greeting, gently acknowledging any open struggles from past sessions.
        """
        if self._closed:
            return
        await self.ensure_connected()
        if not self._session:
            return

        greeting_prompt = "[System event: The session is beginning. "
        if user_name:
            greeting_prompt += f"The person's name is {user_name}. "
        if unresolved_episodes and len(unresolved_episodes) > 0:
            greeting_prompt += (
                "Warmly welcome the person and compassionately check in on how they are feeling today, "
                "mindfully referencing their open situation without being abrupt or clinical."
            )
        else:
            greeting_prompt += "Warmly welcome the person and invite them to share how they are doing today."
        greeting_prompt += " Please speak your opening greeting now.]"

        try:
            turns = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=greeting_prompt)],
                )
            ]
            await self._session.send_client_content(turns=turns, turn_complete=True)
            logger.info(f"[GeminiLive] Proactive opening greeting triggered for session {self.session_id}")
        except Exception as e:
            logger.warning(f"[GeminiLive] Failed to trigger kickoff greeting: {e}")

    # ── Response Listener ─────────────────────────────────────────────────────

    async def _listen_loop(self) -> None:
        """
        Background task that continuously reads Gemini Live responses
        and dispatches them as structured events via self.on_event().
        """
        accumulated_text = ""
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
                        self._resume_handle = handle
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

                    # Tool call handling from Gemini Live agent
                    tool_call = getattr(response, "tool_call", None)
                    if tool_call and getattr(tool_call, "function_calls", None):
                        function_responses = []
                        for call in tool_call.function_calls:
                            call_name = getattr(call, "name", "")
                            call_id = getattr(call, "id", "")
                            call_args = getattr(call, "args", {}) or {}
                            logger.info(
                                f"[GeminiLive] Tool call invoked: {call_name} (id={call_id}) args={call_args}"
                            )
                            try:
                                dispatch_res = await tool_dispatcher.dispatch(
                                    tool_name=call_name,
                                    args=call_args,
                                    context={
                                        "user_id": self.user_id,
                                        "session_id": self.session_id,
                                        "auth_token": self.auth_token,
                                    },
                                )
                                result_payload = dispatch_res.get("result", dispatch_res)
                                function_responses.append(
                                    types.FunctionResponse(
                                        name=call_name,
                                        id=call_id,
                                        response=result_payload if isinstance(result_payload, dict) else {"output": result_payload},
                                    )
                                )
                            except Exception as call_err:
                                logger.error(f"[GeminiLive] Failed executing tool {call_name}: {call_err}", exc_info=True)
                                function_responses.append(
                                    types.FunctionResponse(
                                        name=call_name,
                                        id=call_id,
                                        response={"error": str(call_err)},
                                    )
                                )

                        if function_responses:
                            await self._session.send_tool_response(function_responses=function_responses)
                            logger.info(f"[GeminiLive] Sent {len(function_responses)} tool response(s) back to Gemini Live")
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
                                # Text output (accumulate streaming chunks for text modality)
                                if part.text:
                                    accumulated_text += part.text
                                    await self.on_event({
                                        "type": "model_transcript",
                                        "text": accumulated_text,
                                        "is_final": False,
                                    })

                        # Output transcription (model speech-to-text live stream)
                        output_transcription = getattr(
                            server_content, "output_transcription", None
                        )
                        if output_transcription and output_transcription.text:
                            accumulated_text += output_transcription.text
                            await self.on_event({
                                "type": "model_transcript",
                                "text": accumulated_text,
                                "is_final": False,
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

                        # Generation complete (text generation finished; audio stream may still be playing)
                        # Finalize model transcript immediately so quick reloads persist the AI response
                        if getattr(server_content, "generation_complete", False):
                            if accumulated_text and accumulated_text.strip():
                                final_model_text = accumulated_text.strip()
                                accumulated_text = ""
                                await self.on_event({
                                    "type": "model_transcript",
                                    "text": final_model_text,
                                    "is_final": True,
                                })

                        # Interruption / barge-in
                        if getattr(server_content, "interrupted", False):
                            interrupted_text = accumulated_text.strip()
                            accumulated_text = ""
                            await self.on_event({
                                "type": "interrupted",
                                "text": interrupted_text,
                                "timestamp_ms": int(time.time() * 1000),
                            })

                        # Turn complete
                        if getattr(server_content, "turn_complete", False):
                            # Fallback: if not already finalized on generation_complete, finalize now
                            if accumulated_text and accumulated_text.strip():
                                final_model_text = accumulated_text.strip()
                                accumulated_text = ""
                                await self.on_event({
                                    "type": "model_transcript",
                                    "text": final_model_text,
                                    "is_final": True,
                                })

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
            # Flush any unpersisted model text so disconnecting/reloading never loses the turn
            if accumulated_text and accumulated_text.strip():
                try:
                    await self.on_event({
                        "type": "model_transcript",
                        "text": accumulated_text.strip(),
                        "is_final": True,
                    })
                except Exception as flush_err:
                    logger.warning(f"[GeminiLive] Error flushing accumulated text on cancel: {flush_err}")
                accumulated_text = ""
        except Exception as e:
            if not self._closed:
                err_str = str(e).lower()
                is_inactivity = (
                    "1008" in err_str
                    or "inactivity" in err_str
                    or "closed due to client inactivity" in err_str
                )
                if is_inactivity:
                    logger.info(
                        f"[GeminiLive] Session {self.session_id} paused due to client inactivity. Will resume automatically on next input."
                    )
                    self._session = None
                    self._session_cm = None
                    await self.on_event({
                        "type": "session_idle",
                        "message": "Session idle due to inactivity",
                    })
                    return

                logger.error(
                    f"[GeminiLive] Listener error for session {self.session_id}: {e}",
                    exc_info=True,
                )
                if accumulated_text and accumulated_text.strip():
                    try:
                        await self.on_event({
                            "type": "model_transcript",
                            "text": accumulated_text.strip(),
                            "is_final": True,
                        })
                    except Exception as flush_err:
                        logger.warning(f"[GeminiLive] Error flushing accumulated text on error: {flush_err}")
                    accumulated_text = ""
                await self.on_event({
                    "type": "error",
                    "message": str(e),
                })

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def close(self) -> None:
        """Cleanly shuts down the Gemini Live session, listener, and keepalive."""
        if self._closed:
            return
        self._closed = True

        if self._keepalive_task and not self._keepalive_task.done():
            self._keepalive_task.cancel()
            try:
                await self._keepalive_task
            except asyncio.CancelledError:
                pass

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
