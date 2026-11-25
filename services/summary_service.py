"""
Rolling Conversation Summary Service
====================================
Maintains a rolling clinical summary of the therapy session.
Applies a cache write-back policy:
1. Generates an updated summary using a lightweight LLM (gemini-3.5-flash-lite).
2. Updates Redis cache immediately for low-latency retrieval.
3. Asynchronously writes back the updated summary to MongoDB via Node backend.
"""

import asyncio
import logging
from typing import Optional
from google import genai
from core.config import settings
from prompts.summary import build_summary_prompt
from services.redis_service import redis_service
from services.node_client import node_client

logger = logging.getLogger(__name__)


class SummaryService:
    """
    Manages generation, caching, and database write-back of rolling session summaries.
    """

    def __init__(self):
        self._client: Optional[genai.Client] = None

    def _get_client(self) -> Optional[genai.Client]:
        if self._client is None:
            api_key = settings.GEMINI_API_KEY
            if not api_key:
                logger.warning("[SummaryService] No GEMINI_API_KEY configured for summarization.")
                return None
            self._client = genai.Client(api_key=api_key)
        return self._client

    def generate_summary_sync(
        self, existing_summary: str, user_text: str, model_text: str
    ) -> str:
        """
        Synchronously calls Gemini to generate an updated rolling summary.
        Kept synchronous so it can run via asyncio.to_thread without blocking event loop.
        """
        client = self._get_client()
        if not client:
            return existing_summary

        fallback_summary = existing_summary.strip()
        prompt = build_summary_prompt(
            existing_summary=fallback_summary,
            user_text=user_text,
            model_text=model_text,
        )

        try:
            response = client.models.generate_content(
                model=settings.GEMINI_TEXT_MODEL,
                contents=prompt,
            )

            raw_text = ""
            if (
                response.candidates
                and response.candidates[0].content
                and response.candidates[0].content.parts
            ):
                text_parts = [
                    p.text
                    for p in response.candidates[0].content.parts
                    if getattr(p, "text", None)
                ]
                raw_text = "".join(text_parts).strip()

            if raw_text:
                logger.info(
                    f"[SummaryService] Generated updated summary ({len(raw_text)} chars, ~{len(raw_text.split())} words)"
                )
                return raw_text
            else:
                logger.warning("[SummaryService] Empty response from Gemini summary model.")
                return fallback_summary

        except Exception as e:
            logger.error(f"[SummaryService] Summarization failed: {e}", exc_info=True)
            return fallback_summary

    async def update_summary_async(
        self,
        session_id: str,
        user_text: str,
        model_text: str,
        auth_token: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        Asynchronously updates the rolling summary and applies cache write-back policy:
        1. Fetch current summary from Redis (or fallback to MongoDB).
        2. Call LLM in threadpool to produce updated summary.
        3. Write updated summary to Redis immediately.
        4. Write back to MongoDB via Node backend.
        """
        if not session_id or not (user_text or model_text):
            return None

        try:
            # 1. Retrieve current summary from Redis
            existing_summary = await redis_service.get_session_summary(session_id)
            if existing_summary is None:
                # Rehydrate from MongoDB via Node backend if not in Redis
                session_data = await node_client.fetch_session_data(
                    session_id=session_id, auth_token=auth_token, user_id=user_id
                )
                if session_data and isinstance(session_data, dict):
                    existing_summary = session_data.get("conversationSummary") or ""
                else:
                    existing_summary = ""

            # 2. Generate updated summary off the async event loop
            updated_summary = await asyncio.to_thread(
                self.generate_summary_sync,
                existing_summary,
                user_text,
                model_text,
            )

            if not updated_summary:
                return existing_summary

            # 3. Cache Write-Back Policy:
            # Step A: Update Redis cache immediately
            await redis_service.set_session_summary(session_id, updated_summary)
            logger.debug(f"[SummaryService] Cached updated summary in Redis for session={session_id}")

            # Step B: Asynchronously persist to MongoDB via Node backend
            persisted = await node_client.patch_session_summary(
                session_id=session_id,
                summary=updated_summary,
                auth_token=auth_token,
                user_id=user_id,
            )
            if persisted:
                logger.info(f"[SummaryService] Successfully write-backed summary to MongoDB for session={session_id}")
            else:
                logger.warning(f"[SummaryService] Cache write-back to MongoDB failed for session={session_id}")

            return updated_summary

        except Exception as e:
            logger.error(
                f"[SummaryService] Unexpected error in update_summary_async for session {session_id}: {e}",
                exc_info=True,
            )
            return None


summary_service = SummaryService()
