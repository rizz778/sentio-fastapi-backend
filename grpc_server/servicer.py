import asyncio
import logging
import time
from typing import Any, AsyncIterable, Dict, Optional

from proto import sentio_gateway_pb2 as pb2
from proto import sentio_gateway_pb2_grpc as pb2_grpc
from core.config import settings
from prompts.chat import build_chat_system_instruction
from services.gemini_live_service import GeminiLiveSession
from services.redis_service import redis_service
from services.node_client import node_client
from services.turn_processor import turn_processor
from services.emotion_service import emotion_service

logger = logging.getLogger(__name__)


class SentioLiveGatewayServicer(pb2_grpc.SentioLiveGatewayServicer):
    """
    gRPC Servicer implementing SentioLiveGateway bidirectional stream.
    Orchestrates:
    - Gateway ClientEvent stream (SessionInit, RealtimeAudio, ClientInterrupt, ClientPing)
    - Unified GeminiLiveSession (voice, client modalities, transcripts, resume handles)
    - Redis sliding window & emotion caching
    - TurnProcessor (distress telemetry pushed to client, persistence to Node backend)
    """

    async def StreamSession(
        self,
        request_iterator: AsyncIterable[pb2.ClientEvent],
        context,
    ) -> AsyncIterable[pb2.ServerEvent]:
        outbound_queue: asyncio.Queue[Optional[pb2.ServerEvent]] = asyncio.Queue()
        session_id: Optional[str] = None
        user_id: Optional[str] = None
        auth_token: Optional[str] = None
        gemini_session: Optional[GeminiLiveSession] = None
        session_initialized = False

        async def on_distress_ready(distress_score: float, emotion: str, timestamp_ms: int) -> None:
            """Callback from TurnProcessor when distress scoring completes for a user turn."""
            distress_event = pb2.ServerEvent(
                distress_update=pb2.DistressUpdate(
                    score=distress_score,
                    emotion=emotion,
                    timestamp_ms=timestamp_ms,
                )
            )
            await outbound_queue.put(distress_event)

        async def handle_gemini_event(event: Dict[str, Any]) -> None:
            """Translates Gemini Live events to protobuf ServerEvents and dispatches them."""
            event_type = event.get("type")

            if event_type == "audio_chunk":
                audio_event = pb2.ServerEvent(
                    audio_chunk=pb2.RealtimeAudio(
                        data=event["data"],
                        mime_type=event.get("mime_type", "audio/pcm;rate=24000"),
                    )
                )
                await outbound_queue.put(audio_event)

            elif event_type == "user_transcript":
                text = event.get("text", "")
                is_final = event.get("is_final", True)
                transcript_event = pb2.ServerEvent(
                    transcript=pb2.Transcript(
                        speaker=pb2.Transcript.USER,
                        text=text,
                        is_final=is_final,
                        timestamp_ms=int(time.time() * 1000),
                    )
                )
                await outbound_queue.put(transcript_event)

                # On final user transcript, trigger non-blocking turn processing & distress evaluation
                if is_final and text.strip() and session_id and user_id:
                    turn_processor.launch_user_turn(
                        session_id=session_id,
                        user_id=user_id,
                        text=text,
                        on_distress_ready=on_distress_ready,
                        auth_token=auth_token,
                    )

            elif event_type == "model_transcript":
                text = event.get("text", "")
                is_final = event.get("is_final", False)
                transcript_event = pb2.ServerEvent(
                    transcript=pb2.Transcript(
                        speaker=pb2.Transcript.MODEL,
                        text=text,
                        is_final=is_final,
                        timestamp_ms=int(time.time() * 1000),
                    )
                )
                await outbound_queue.put(transcript_event)

                # On final model transcript, persist turn asynchronously
                if is_final and text.strip() and session_id:
                    turn_processor.launch_model_turn(
                        session_id=session_id,
                        text=text,
                        user_id=user_id,
                        auth_token=auth_token,
                    )

            elif event_type in ("turn_complete", "generation_complete"):
                # Forward turn completion signal so client releases input lock immediately
                turn_done_event = pb2.ServerEvent(
                    transcript=pb2.Transcript(
                        speaker=pb2.Transcript.MODEL,
                        text="",
                        is_final=True,
                        timestamp_ms=int(time.time() * 1000),
                    )
                )
                await outbound_queue.put(turn_done_event)

            elif event_type == "interrupted":
                stop_event = pb2.ServerEvent(
                    stop_playback=pb2.StopPlayback(
                        timestamp_ms=event.get("timestamp_ms", int(time.time() * 1000)),
                        reason="gemini_interrupted",
                    )
                )
                await outbound_queue.put(stop_event)

            elif event_type == "resume_handle":
                handle = event.get("handle")
                if handle and session_id:
                    await redis_service.set_gemini_resume_handle(session_id, handle)
                    logger.debug(f"[gRPC] Cached new Gemini resume handle for {session_id}")

            elif event_type == "error":
                err_event = pb2.ServerEvent(
                    error=pb2.GatewayError(
                        code=pb2.GatewayError.INTERNAL_ERROR,
                        message=event.get("message", "Gemini Live error"),
                        recoverable=True,
                    )
                )
                await outbound_queue.put(err_event)

        async def inbound_reader():
            nonlocal session_id, user_id, auth_token, gemini_session, session_initialized
            try:
                async for client_event in request_iterator:
                    payload_field = client_event.WhichOneof("payload")

                    if payload_field == "session_init":
                        init = client_event.session_init
                        session_id = init.session_id
                        user_id = init.user_id
                        auth_token = init.auth_token

                        # Modalities come from client (default AUDIO); voice and instructions come from FastAPI config
                        response_modalities = (
                            list(init.response_modalities) if init.response_modalities else ["AUDIO"]
                        )
                        voice_name = settings.GEMINI_VOICE_NAME

                        logger.info(
                            f"[gRPC] SessionInit: session_id={session_id}, user_id={user_id}, "
                            f"modalities={response_modalities}, voice={voice_name}"
                        )

                        # 1. Parallel fetch/cache of user profile and session history
                        profile_task = asyncio.create_task(
                            self._resolve_user_profile(user_id, auth_token, session_id)
                        )
                        history_task = asyncio.create_task(
                            self._resolve_session_history(session_id, auth_token, user_id)
                        )
                        resume_handle_task = asyncio.create_task(
                            redis_service.get_gemini_resume_handle(session_id)
                        )

                        user_profile, prior_messages, resume_handle = await asyncio.gather(
                            profile_task, history_task, resume_handle_task
                        )

                        # 2. Build system instruction from FastAPI config and user profile
                        system_instruction = build_chat_system_instruction(
                            user_profile=user_profile,
                            conversation_history=prior_messages,
                            custom_instruction=settings.DEFAULT_SYSTEM_INSTRUCTION,
                        )

                        # 3. Instantiate and connect Gemini Live session
                        gemini_session = GeminiLiveSession(
                            session_id=session_id,
                            user_id=user_id,
                            on_event=handle_gemini_event,
                        )

                        # Connect opens session, applies client config, and injects Redis history.
                        # If a cached resumption handle has expired on Google's end, clear it and retry fresh.
                        try:
                            await gemini_session.connect(
                                system_instruction=system_instruction,
                                voice_name=voice_name,
                                response_modalities=response_modalities,
                                resume_handle=resume_handle,
                            )
                        except Exception as conn_err:
                            err_str = str(conn_err).lower()
                            if resume_handle and ("not found" in err_str or "expired" in err_str or "1008" in err_str):
                                logger.warning(
                                    f"[gRPC] Gemini session resumption failed ({conn_err}), clearing stale handle and connecting fresh..."
                                )
                                await redis_service.clear_gemini_resume_handle(session_id)
                                await gemini_session.connect(
                                    system_instruction=system_instruction,
                                    voice_name=voice_name,
                                    response_modalities=response_modalities,
                                    resume_handle=None,
                                )
                            else:
                                raise conn_err

                        session_initialized = True

                        # 4. Acknowledge readiness to Gateway SessionBridge
                        ready_event = pb2.ServerEvent(
                            session_ready=pb2.SessionReady(
                                session_id=session_id,
                                prior_turn_count=len(prior_messages) if prior_messages else 0,
                                rehydrated=bool(resume_handle),
                                message="FastAPI Sentio Live session initialized",
                            )
                        )
                        await outbound_queue.put(ready_event)

                    elif payload_field == "audio_chunk":
                        audio = client_event.audio_chunk
                        if gemini_session:
                            await gemini_session.send_audio(
                                audio.data,
                                audio.mime_type or "audio/pcm;rate=16000",
                            )

                    elif payload_field == "interrupt":
                        interrupt = client_event.interrupt
                        logger.info(
                            f"[gRPC] Client barge-in interrupt: {interrupt.reason}"
                        )
                        stop_event = pb2.ServerEvent(
                            stop_playback=pb2.StopPlayback(
                                timestamp_ms=interrupt.timestamp_ms or int(time.time() * 1000),
                                reason=interrupt.reason or "user_barge_in",
                            )
                        )
                        await outbound_queue.put(stop_event)

                    elif payload_field == "ping":
                        ping = client_event.ping
                        pong_event = pb2.ServerEvent(
                            pong=pb2.ServerPong(
                                timestamp_ms=ping.timestamp_ms or int(time.time() * 1000)
                            )
                        )
                        await outbound_queue.put(pong_event)

                    elif payload_field == "video_frame":
                        video_frame = client_event.video_frame
                        if video_frame.data and user_id:
                            # Asynchronously process video frame via hosted vision model without blocking audio stream
                            asyncio.create_task(
                                self._handle_video_frame(
                                    session_id=session_id,
                                    user_id=user_id,
                                    frame_data=video_frame.data,
                                    mime_type=video_frame.mime_type or "image/jpeg",
                                    gemini_session=gemini_session,
                                    outbound_queue=outbound_queue,
                                )
                            )

                    elif payload_field == "text_message":
                        text_msg = client_event.text_message
                        user_text = text_msg.text.strip()
                        if user_text and session_id and user_id:
                            logger.info(
                                f"[gRPC] User typed text turn: session={session_id} text={user_text[:80]}"
                            )
                            # 1. Forward typed text turn into Gemini Live session
                            if gemini_session:
                                await gemini_session.send_text(user_text)

                            # 2. Launch non-blocking user turn processing (distress scoring, Redis window, MongoDB persistence)
                            turn_processor.launch_user_turn(
                                session_id=session_id,
                                user_id=user_id,
                                text=user_text,
                                on_distress_ready=on_distress_ready,
                                auth_token=auth_token,
                            )

                    else:
                        logger.warning(f"[gRPC] Unhandled client payload: {payload_field}")

            except asyncio.CancelledError:
                logger.info(f"[gRPC] Inbound stream reader cancelled for session={session_id}")
            except Exception as e:
                logger.error(
                    f"[gRPC] Error in inbound stream reader for session={session_id}: {e}",
                    exc_info=True,
                )
                err_event = pb2.ServerEvent(
                    error=pb2.GatewayError(
                        code=pb2.GatewayError.INTERNAL_ERROR,
                        message=str(e),
                        recoverable=False,
                    )
                )
                await outbound_queue.put(err_event)
            finally:
                # Sentinel to end outbound stream
                await outbound_queue.put(None)

        reader_task = asyncio.create_task(inbound_reader())

        try:
            while True:
                event = await outbound_queue.get()
                if event is None:
                    break
                yield event
        except asyncio.CancelledError:
            logger.info(f"[gRPC] StreamSession cancelled for session={session_id}")
        finally:
            if not reader_task.done():
                reader_task.cancel()
                try:
                    await reader_task
                except asyncio.CancelledError:
                    pass

            if gemini_session:
                await gemini_session.close()

            if session_id:
                emotion_service.cleanup_session(session_id)

            logger.info(f"[gRPC] StreamSession terminated for session={session_id}")

    async def _resolve_user_profile(
        self, user_id: str, auth_token: Optional[str], session_id: str
    ) -> Optional[Dict[str, Any]]:
        """Checks Redis for cached profile; falls back to Node backend."""
        profile = await redis_service.get_session_profile(session_id)
        if profile:
            return profile

        profile = await node_client.fetch_user_profile(user_id=user_id, auth_token=auth_token)
        if profile:
            await redis_service.set_session_profile(session_id, profile)
        return profile

    async def _resolve_session_history(
        self, session_id: str, auth_token: Optional[str], user_id: Optional[str]
    ) -> list:
        """Checks Redis for sliding window; falls back to Node backend to rehydrate."""
        window = await redis_service.get_sliding_window(session_id)
        if window:
            return window

        # Failover / initial recovery: fetch from Node MongoDB
        messages = await node_client.fetch_session_messages(
            session_id=session_id, auth_token=auth_token, user_id=user_id
        )
        if messages:
            # Seed the sliding window in Redis with up to 5 latest messages
            for msg in messages[-5:]:
                await redis_service.push_sliding_message(session_id, msg)
            return messages[-5:]
        return []

    async def _handle_video_frame(
        self,
        session_id: str,
        user_id: str,
        frame_data: bytes,
        mime_type: str,
        gemini_session: Optional[GeminiLiveSession],
        outbound_queue: asyncio.Queue,
    ) -> None:
        """
        Runs hosted async Gemini Flash emotion recognition, caches running emotion to Redis,
        injects context cue into Gemini Live if emotion shifted,
        and pushes real-time DistressUpdate to Gateway/client dashboard.
        """
        try:
            current_emotion, aggregated_emotion = await emotion_service.analyze_frame_bytes_async(
                frame_data, mime_type=mime_type, session_id=session_id
            )

            # Check previous running emotion to detect shifts
            prev_emotion_data = await redis_service.get_user_emotion(user_id)
            prev_emotion = prev_emotion_data.get("emotion") if prev_emotion_data else None

            # Always update Redis with latest smoothed running emotion
            await redis_service.set_user_emotion(user_id, aggregated_emotion)

            # If user emotion shifted, silently cue Gemini Live to adapt vocal empathy
            if prev_emotion and prev_emotion != aggregated_emotion and gemini_session:
                await gemini_session.inject_emotion_context(aggregated_emotion)

            # Push real-time emotion telemetry to client dashboard
            telemetry_event = pb2.ServerEvent(
                distress_update=pb2.DistressUpdate(
                    score=0.0,
                    emotion=aggregated_emotion,
                    timestamp_ms=int(time.time() * 1000),
                )
            )
            await outbound_queue.put(telemetry_event)

        except Exception as e:
            logger.warning(f"[gRPC] Error handling video frame for session={session_id}: {e}")
