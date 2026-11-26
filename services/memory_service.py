"""
Consolidated Memory Service
===========================
Orchestrates single-pass long-term memory operations after each model turn:
1. Generates rolling clinical session summary.
2. Salience Gatekeeper extracts episodic memories (conflicts, breakthroughs, spikes).
3. Generates vector embeddings via Google GenAI (text-embedding-004).
4. Persists rolling summary to Redis cache and MongoDB.
5. Persists salient structured episodes to MongoDB Atlas Vector Index via MemoryClient.
"""

import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Optional
from google import genai
from google.genai import types
from core.config import settings
from prompts.memory_consolidation import build_memory_consolidation_prompt
from services.redis_service import redis_service
from clients.node.session import session_client
from clients.node.memory import memory_client

logger = logging.getLogger(__name__)


class MemoryService:
    """
    Manages session summaries and episodic memory extraction, embedding, and retrieval.
    """

    def __init__(self):
        self._client: Optional[genai.Client] = None

    def _get_client(self) -> Optional[genai.Client]:
        """Lazy initialization of the Google GenAI client with explicit API version."""
        if self._client is None:
            api_key = settings.GEMINI_API_KEY
            if not api_key:
                logger.warning("[MemoryService] GEMINI_API_KEY is not set.")
                return None
            api_version = getattr(settings, "GEMINI_API_VERSION", "v1beta") or "v1beta"
            self._client = genai.Client(
                api_key=api_key,
                http_options=types.HttpOptions(api_version=api_version),
            )
        return self._client

    # ── Vector Embeddings ───────────────────────────────────────────────────────

    def embed_text(self, text: str) -> List[float]:
        """
        Synchronously generates a 768-dimensional float embedding using Gemini embedding model.
        """
        if not text or not text.strip():
            return []

        client = self._get_client()
        if not client:
            return []

        target_model = getattr(settings, "GEMINI_EMBEDDING_MODEL", "gemini-embedding-001") or "gemini-embedding-001"
        candidate_models = [target_model, "gemini-embedding-001", "models/gemini-embedding-001"]
        # Remove duplicates while preserving order
        candidate_models = list(dict.fromkeys(candidate_models))

        for model_name in candidate_models:
            # 1. Try with output_dimensionality=768 for compact vector indexing
            try:
                response = client.models.embed_content(
                    model=model_name,
                    contents=text.strip(),
                    config=types.EmbedContentConfig(output_dimensionality=768),
                )
                if response.embeddings and len(response.embeddings) > 0:
                    values = response.embeddings[0].values
                    if values and isinstance(values, list):
                        logger.debug(f"[MemoryService] Generated {len(values)}-dim embedding with '{model_name}'")
                        return values
            except Exception as e:
                logger.debug(f"[MemoryService] Configured embedding attempt with '{model_name}' failed: {e}")

            # 2. Fallback without config
            try:
                response = client.models.embed_content(
                    model=model_name,
                    contents=text.strip(),
                )
                if response.embeddings and len(response.embeddings) > 0:
                    values = response.embeddings[0].values
                    if values and isinstance(values, list):
                        logger.debug(f"[MemoryService] Generated standard embedding with '{model_name}'")
                        return values
            except Exception as e:
                logger.warning(
                    f"[MemoryService] Embedding generation attempt with '{model_name}' failed: {e}"
                )

        logger.error(f"[MemoryService] All embedding attempts failed for text ({len(text)} chars)")
        return []

    async def embed_text_async(self, text: str) -> List[float]:
        """Asynchronous wrapper for vector embedding generation."""
        return await asyncio.to_thread(self.embed_text, text)

    # ── Consolidated Memory Generation ──────────────────────────────────────────

    def generate_consolidation_sync(
        self,
        existing_summary: str,
        user_text: str,
        model_text: str,
        recent_context: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Executes single LLM call for both rolling summary and episodic gatekeeper.
        """
        client = self._get_client()
        if not client:
            return {"updatedSummary": existing_summary, "salientEpisode": None}

        prompt = build_memory_consolidation_prompt(
            existing_summary=existing_summary,
            user_text=user_text,
            model_text=model_text,
            recent_context=recent_context,
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
            if not raw_text:
                raw_text = (getattr(response, "text", None) or "").strip()

            cleaned = re.sub(r"^```(?:json)?|```$", "", raw_text.strip(), flags=re.MULTILINE).strip()

            try:
                data = json.loads(cleaned)
                if isinstance(data, dict):
                    return data
            except Exception as json_err:
                logger.warning(
                    f"[MemoryService] JSON parsing failed ({json_err}). Extracting text fallback."
                )

            # Fallback if strict JSON parsing fails
            return {
                "updatedSummary": cleaned if len(cleaned) < 1000 else existing_summary,
                "salientEpisode": None,
            }

        except Exception as e:
            logger.error(f"[MemoryService] Memory consolidation call failed: {e}", exc_info=True)
            return {"updatedSummary": existing_summary, "salientEpisode": None}

    async def consolidate_memory_async(
        self,
        session_id: str,
        user_text: str,
        model_text: str,
        auth_token: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Consolidates long-term memory after a model response turn:
        1. Fetches current session summary from Redis (or MongoDB fallback).
        2. Retrieves recent conversation window.
        3. Executes consolidated LLM call in a background thread.
        4. Writes updated rolling summary to Redis and persists back to MongoDB.
        5. If salient episode detected, computes vector embedding and creates episode record.
        """
        try:
            # 1. Fetch existing summary (Redis -> Mongo fallback)
            existing_summary = await redis_service.get_session_summary(session_id)
            if not existing_summary:
                session_data = await session_client.fetch_session_data(
                    session_id=session_id, auth_token=auth_token, user_id=user_id
                )
                if session_data and isinstance(session_data, dict):
                    existing_summary = session_data.get("conversationSummary") or ""

            existing_summary = (existing_summary or "").strip()

            # 2. Fetch recent conversation messages for context
            recent_context = await redis_service.get_recent_messages(session_id, count=6)

            # 3. Call LLM for consolidated evaluation
            result = await asyncio.to_thread(
                self.generate_consolidation_sync,
                existing_summary,
                user_text,
                model_text,
                recent_context,
            )

            # 4. Handle updated rolling summary
            updated_summary = (result.get("updatedSummary") or "").strip()
            if updated_summary and updated_summary != existing_summary:
                # Cache write-back: update Redis first
                await redis_service.set_session_summary(session_id, updated_summary)
                # Persist to MongoDB
                await session_client.patch_session_summary(
                    session_id=session_id,
                    summary=updated_summary,
                    auth_token=auth_token,
                    user_id=user_id,
                )
                logger.info(
                    f"[MemoryService] Session summary updated for session {session_id} (~{len(updated_summary.split())} words)"
                )

            # 5. Handle salient episodic memory
            salient_episode = result.get("salientEpisode")
            if (
                isinstance(salient_episode, dict)
                and salient_episode.get("isSalient")
                and user_id
            ):
                event_summary = (salient_episode.get("eventSummary") or "").strip()
                if event_summary:
                    # Generate 768-dim vector embedding of the event summary
                    embedding = await self.embed_text_async(event_summary)
                    if embedding:
                        salient_episode["embedding"] = embedding

                    # Persist structured episode to MongoDB via MemoryClient
                    created_ep = await memory_client.create_episode(
                        user_id=user_id,
                        session_id=session_id,
                        episode_data=salient_episode,
                        auth_token=auth_token,
                    )
                    if created_ep:
                        logger.info(
                            f"[MemoryService] 💾 Salient episode stored for user {user_id} ({salient_episode.get('eventType')}): '{event_summary[:50]}...'"
                        )

            return result

        except Exception as e:
            logger.error(
                f"[MemoryService] Error consolidating memory for session {session_id}: {e}",
                exc_info=True,
            )
            return {"updatedSummary": "", "salientEpisode": None}

    # ── Retrieval Operations ───────────────────────────────────────────────────

    async def query_episodes(
        self,
        user_id: str,
        query_text: str,
        limit: int = 3,
        auth_token: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Embeds a text query and performs vector similarity search across user episodes.
        """
        if not user_id or not query_text:
            return []

        query_vector = await self.embed_text_async(query_text)
        if not query_vector:
            logger.warning("[MemoryService] Could not generate embedding for search query.")
            return []

        return await memory_client.search_episodes(
            user_id=user_id,
            query_vector=query_vector,
            limit=limit,
            auth_token=auth_token,
        )

    async def recall_unresolved_episodes(
        self,
        user_id: str,
        limit: int = 5,
        auth_token: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Fetches unresolved episodic memories for proactive session-start recall.
        """
        if not user_id:
            return []
        return await memory_client.fetch_user_episodes(
            user_id=user_id,
            status="unresolved",
            limit=limit,
            auth_token=auth_token,
        )


memory_service = MemoryService()
