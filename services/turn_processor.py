"""
Turn Processor
==============
Orchestrates turn-level lifecycle operations across user and model messages:
- User turn: snapshots emotion from Redis, queries side LLM for distress,
  appends to Redis sliding window, fires async persist to Node, emits distress telemetry.
- Model turn: appends AI response to Redis sliding window, fires async persist to Node.

Kept strictly separate from LLM scoring logic in distress_service.py.
"""

import asyncio
import datetime
import logging
import time
from typing import Any, Callable, Coroutine, Dict, Optional

from services.redis_service import redis_service
from services.distress_service import distress_service
from services.memory_service import memory_service
from clients.node.session import session_client as node_client

logger = logging.getLogger(__name__)


class TurnProcessor:
    """
    Handles turn orchestration for both user speech/text and AI model responses.
    """

    async def process_user_turn(
        self,
        session_id: str,
        user_id: str,
        text: str,
        on_distress_ready: Callable[[float, str, int], Coroutine[Any, Any, None]],
        auth_token: Optional[str] = None,
    ) -> None:
        """
        Processes a completed user turn.
        Runs as a background task to keep the live audio stream non-blocking.
        """
        try:
            timestamp_ms = int(time.time() * 1000)
            user_iso_timestamp = datetime.datetime.fromtimestamp(
                timestamp_ms / 1000.0, datetime.timezone.utc
            ).isoformat()

            # 1. Get current sliding window from Redis for context
            sliding_window = await redis_service.get_sliding_window(session_id)

            # 2. Compute text distress score & categorized emotion asynchronously in a single call
            assessment = await distress_service.assess_turn(text, sliding_window)
            distress_score = assessment.get("distressScore", 0.0)
            emotion = assessment.get("emotion", "neutral")

            # 3. Persist user message to MongoDB with text-evaluated emotion and distress score
            node_client.fire_and_forget_persist(
                session_id=session_id,
                sender="user",
                text=text,
                emotion=emotion,
                distress_score=distress_score,
                auth_token=auth_token,
                user_id=user_id,
                timestamp=user_iso_timestamp,
            )

            # 4. Push user message with distress score and emotion into Redis sliding window
            enriched_user_msg = {
                "sender": "user",
                "text": text,
                "emotion": emotion,
                "distressScore": distress_score,
                "timestamp": timestamp_ms,
            }
            await redis_service.push_sliding_message(session_id, enriched_user_msg)

            # 5. Notify callback to push DistressUpdate over gRPC to client
            await on_distress_ready(distress_score, emotion, timestamp_ms)

            logger.info(
                f"[TurnProcessor] User turn processed: session={session_id} "
                f"user={user_id} emotion={emotion} distress={distress_score:.1f}"
            )

        except Exception as e:
            logger.error(
                f"[TurnProcessor] Error processing user turn for session={session_id}: {e}",
                exc_info=True,
            )

    async def process_model_turn(
        self,
        session_id: str,
        text: str,
        user_id: Optional[str] = None,
        auth_token: Optional[str] = None,
    ) -> None:
        """
        Processes a completed model turn (persists to Redis sliding window and Node backend).
        """
        try:
            timestamp_ms = int(time.time() * 1000)
            model_iso_timestamp = datetime.datetime.fromtimestamp(
                timestamp_ms / 1000.0, datetime.timezone.utc
            ).isoformat()

            # Push model response into Redis sliding window
            model_msg = {
                "sender": "model",
                "text": text,
                "timestamp": timestamp_ms,
            }
            await redis_service.push_sliding_message(session_id, model_msg)

            # Non-blocking persist to MongoDB via Node backend with explicit model timestamp
            node_client.fire_and_forget_persist(
                session_id=session_id,
                sender="model",
                text=text,
                auth_token=auth_token,
                user_id=user_id,
                timestamp=model_iso_timestamp,
            )

            # Find the most recent user turn from sliding window to pair for summarization
            recent_msgs = await redis_service.get_recent_messages(session_id, count=4)
            last_user_text = ""
            for msg in reversed(recent_msgs):
                if isinstance(msg, dict) and msg.get("sender") == "user":
                    last_user_text = msg.get("text", "")
                    break

            # Trigger asynchronous consolidated memory consolidation (session summary + episodic memory gatekeeper)
            if last_user_text or text:
                asyncio.create_task(
                    memory_service.consolidate_memory_async(
                        session_id=session_id,
                        user_text=last_user_text,
                        model_text=text,
                        auth_token=auth_token,
                        user_id=user_id,
                    )
                )

            logger.debug(
                f"[TurnProcessor] Model turn persisted & memory consolidation queued for session={session_id}"
            )

        except Exception as e:
            logger.error(
                f"[TurnProcessor] Error processing model turn for session={session_id}: {e}",
                exc_info=True,
            )

    def launch_user_turn(
        self,
        session_id: str,
        user_id: str,
        text: str,
        on_distress_ready: Callable[[float, str, int], Coroutine[Any, Any, None]],
        auth_token: Optional[str] = None,
    ) -> asyncio.Task:
        """Schedules user turn processing as an asynchronous non-blocking task."""
        return asyncio.create_task(
            self.process_user_turn(
                session_id=session_id,
                user_id=user_id,
                text=text,
                on_distress_ready=on_distress_ready,
                auth_token=auth_token,
            )
        )

    def launch_model_turn(
        self,
        session_id: str,
        text: str,
        user_id: Optional[str] = None,
        auth_token: Optional[str] = None,
    ) -> asyncio.Task:
        """Schedules model turn processing as an asynchronous non-blocking task."""
        return asyncio.create_task(
            self.process_model_turn(
                session_id=session_id,
                text=text,
                user_id=user_id,
                auth_token=auth_token,
            )
        )


turn_processor = TurnProcessor()
