import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional
import redis.asyncio as aioredis
from core.config import settings

logger = logging.getLogger(__name__)


class RedisService:
    """
    Manages session state, the 5-message sliding window, running facial emotion telemetry,
    and session resumption handles in Redis (with in-memory fallback for local dev).
    """

    def __init__(self, redis_url: Optional[str] = None):
        self.redis_url = redis_url or settings.REDIS_URL
        self._client: Optional[aioredis.Redis] = None
        self._fallback_memory: Dict[str, Any] = {}
        self._use_fallback = False

    async def get_client(self) -> Optional[aioredis.Redis]:
        if self._use_fallback:
            return None
        if self._client is None:
            try:
                self._client = aioredis.from_url(
                    self.redis_url,
                    encoding="utf-8",
                    decode_responses=True,
                    socket_connect_timeout=1.5,
                    socket_timeout=1.5,
                )
                await asyncio.wait_for(self._client.ping(), timeout=1.5)
                logger.info("[RedisService] Successfully connected to Redis.")
            except Exception as e:
                logger.warning(
                    f"[RedisService] Redis unavailable ({e}). Using in-memory state manager."
                )
                self._use_fallback = True
                self._client = None
        return self._client

    # ── User Running Emotion ───────────────────────────────────────────────────

    async def set_user_emotion(
        self, user_id: str, emotion: str, confidence: float = 1.0
    ) -> None:
        """Stores the latest running emotion detected for a user."""
        key = f"sentio:user:{user_id}:emotion"
        data = {
            "emotion": emotion,
            "confidence": confidence,
            "timestamp": time.time(),
        }
        client = await self.get_client()
        if client:
            try:
                await client.set(key, json.dumps(data), ex=3600)
                return
            except Exception as e:
                logger.warning(f"[RedisService] Failed to set emotion: {e}")
        self._fallback_memory[key] = data

    async def get_user_emotion(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Snapshots the latest running emotion for a user."""
        key = f"sentio:user:{user_id}:emotion"
        client = await self.get_client()
        if client:
            try:
                raw = await client.get(key)
                if raw:
                    return json.loads(raw)
            except Exception as e:
                logger.warning(f"[RedisService] Failed to get emotion: {e}")
        return self._fallback_memory.get(key)

    # ── 5-Message Sliding Window ──────────────────────────────────────────────

    async def push_sliding_message(
        self, session_id: str, message: Dict[str, Any], max_window: int = 5
    ) -> None:
        """Appends a message to the sliding window, keeping at most `max_window` items."""
        key = f"sentio:session:{session_id}:recent_messages"
        raw_msg = json.dumps(message)
        client = await self.get_client()
        if client:
            try:
                async with client.pipeline(transaction=True) as pipe:
                    pipe.rpush(key, raw_msg)
                    pipe.ltrim(key, -max_window, -1)
                    pipe.expire(key, 86400)
                    await pipe.execute()
                return
            except Exception as e:
                logger.warning(f"[RedisService] Failed to push sliding message: {e}")
        
        # In-memory fallback
        buffer = self._fallback_memory.setdefault(key, [])
        buffer.append(message)
        if len(buffer) > max_window:
            self._fallback_memory[key] = buffer[-max_window:]

    async def get_sliding_window(self, session_id: str) -> List[Dict[str, Any]]:
        """Retrieves the recent messages in the sliding window."""
        key = f"sentio:session:{session_id}:recent_messages"
        client = await self.get_client()
        if client:
            try:
                items = await client.lrange(key, 0, -1)
                return [json.loads(item) for item in items]
            except Exception as e:
                logger.warning(f"[RedisService] Failed to get sliding window: {e}")
        return self._fallback_memory.get(key, [])

    # ── Session Profile Cache ─────────────────────────────────────────────────

    async def set_session_profile(
        self, session_id: str, profile: Dict[str, Any], ttl_seconds: int = 86400
    ) -> None:
        key = f"sentio:session:{session_id}:profile"
        client = await self.get_client()
        if client:
            try:
                await client.set(key, json.dumps(profile), ex=ttl_seconds)
                return
            except Exception as e:
                logger.warning(f"[RedisService] Failed to set profile: {e}")
        self._fallback_memory[key] = profile

    async def get_session_profile(self, session_id: str) -> Optional[Dict[str, Any]]:
        key = f"sentio:session:{session_id}:profile"
        client = await self.get_client()
        if client:
            try:
                raw = await client.get(key)
                if raw:
                    return json.loads(raw)
            except Exception as e:
                logger.warning(f"[RedisService] Failed to get profile: {e}")
        return self._fallback_memory.get(key)

    # ── Gemini Live Resumption Handle ─────────────────────────────────────────

    async def set_gemini_resume_handle(
        self, session_id: str, handle: str, ttl_seconds: int = 3600
    ) -> None:
        key = f"sentio:session:{session_id}:resume_handle"
        client = await self.get_client()
        if client:
            try:
                await client.set(key, handle, ex=ttl_seconds)
                return
            except Exception as e:
                logger.warning(f"[RedisService] Failed to set resume handle: {e}")
        self._fallback_memory[key] = handle

    async def get_gemini_resume_handle(self, session_id: str) -> Optional[str]:
        key = f"sentio:session:{session_id}:resume_handle"
        client = await self.get_client()
        if client:
            try:
                return await client.get(key)
            except Exception as e:
                logger.warning(f"[RedisService] Failed to get resume handle: {e}")
        return self._fallback_memory.get(key)

    async def clear_gemini_resume_handle(self, session_id: str) -> None:
        key = f"sentio:session:{session_id}:resume_handle"
        client = await self.get_client()
        if client:
            try:
                await client.delete(key)
                return
            except Exception as e:
                logger.warning(f"[RedisService] Failed to delete resume handle: {e}")
        self._fallback_memory.pop(key, None)

    async def close(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None


redis_service = RedisService()
