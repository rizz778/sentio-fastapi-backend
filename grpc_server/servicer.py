import asyncio
import logging
import time
from typing import AsyncIterable, Optional

from proto import sentio_gateway_pb2 as pb2
from proto import sentio_gateway_pb2_grpc as pb2_grpc
from core.config import settings

logger = logging.getLogger(__name__)


class SentioLiveGatewayServicer(pb2_grpc.SentioLiveGatewayServicer):
    """
    gRPC Servicer implementing SentioLiveGateway bidirectional stream.
    Receives ClientEvents from Gateway SessionBridge, handles session lifecycle,
    and returns ServerEvents (session_ready, audio_chunk, transcript, stop_playback, pong, error).
    """

    async def StreamSession(
        self,
        request_iterator: AsyncIterable[pb2.ClientEvent],
        context,
    ) -> AsyncIterable[pb2.ServerEvent]:
        outbound_queue: asyncio.Queue[Optional[pb2.ServerEvent]] = asyncio.Queue()
        session_id: Optional[str] = None
        user_id: Optional[str] = None
        session_initialized = False

        async def inbound_reader():
            nonlocal session_id, user_id, session_initialized
            try:
                async for client_event in request_iterator:
                    payload_field = client_event.WhichOneof("payload")

                    if payload_field == "session_init":
                        init = client_event.session_init
                        session_id = init.session_id
                        user_id = init.user_id
                        logger.info(
                            f"[gRPC] SessionInit received: session_id={session_id}, user_id={user_id}"
                        )

                        # Acknowledge readiness to Gateway SessionBridge
                        ready_event = pb2.ServerEvent(
                            session_ready=pb2.SessionReady(
                                session_id=session_id,
                                prior_turn_count=0,
                                rehydrated=False,
                                message="FastAPI Sentio Live session initialized",
                            )
                        )
                        await outbound_queue.put(ready_event)
                        session_initialized = True

                    elif payload_field == "audio_chunk":
                        audio = client_event.audio_chunk
                        logger.debug(
                            f"[gRPC] Audio chunk received for session={session_id}: {len(audio.data)} bytes ({audio.mime_type})"
                        )

                    elif payload_field == "interrupt":
                        interrupt = client_event.interrupt
                        logger.info(
                            f"[gRPC] Barge-in interrupt received for session={session_id}, reason={interrupt.reason}"
                        )
                        # Immediately instruct client to stop audio playback
                        stop_event = pb2.ServerEvent(
                            stop_playback=pb2.StopPlayback(
                                timestamp_ms=int(time.time() * 1000),
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

                    else:
                        logger.warning(
                            f"[gRPC] Unknown client event payload: {payload_field}"
                        )

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
                # Sentinel to signal termination of the outbound stream
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
            logger.info(f"[gRPC] StreamSession terminated for session={session_id}")
