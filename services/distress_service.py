"""
Distress Scoring Service
========================
Dedicated service for evaluating emotional distress scores using the side LLM.
Separated from session and turn orchestration.
"""

import asyncio
import logging
import re
from typing import Any, List, Optional
from google import genai
from core.config import settings
from prompts.distress import build_distress_prompt

logger = logging.getLogger(__name__)


class DistressService:
    """
    Evaluates emotional distress scores (0.0 to 10.0) from text and recent context.
    """

    def __init__(self):
        self._client: Optional[genai.Client] = None

    def _get_client(self) -> Optional[genai.Client]:
        if self._client is None:
            api_key = settings.DISTRESS_API_KEY or settings.GEMINI_API_KEY
            if not api_key:
                logger.warning("[DistressService] No API key configured for distress scoring.")
                return None
            self._client = genai.Client(api_key=api_key)
        return self._client

    def score_distress_sync(
        self, text: str, conversation_history: Optional[List[Any]] = None
    ) -> float:
        """Synchronously invokes the side LLM to score distress."""
        client = self._get_client()
        if not client:
            return 0.0

        prompt = build_distress_prompt(text, conversation_history)

        try:
            response = client.models.generate_content(
                model=settings.GEMINI_DISTRESS_MODEL,
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

            match = re.search(r"([0-9]+(?:\.[0-9]+)?)", raw_text)
            if match:
                score = float(match.group(1))
                return max(0.0, min(10.0, score))
            logger.warning(
                f"[DistressService] Could not parse numeric score from output: '{raw_text}'"
            )
            return 0.0
        except Exception as e:
            logger.warning(f"[DistressService] Error assessing distress score: {e}")
            return 0.0

    async def score_distress(
        self, text: str, conversation_history: Optional[List[Any]] = None
    ) -> float:
        """
        Asynchronously evaluates distress score via thread pool
        to avoid blocking the asyncio event loop.
        """
        return await asyncio.to_thread(self.score_distress_sync, text, conversation_history)


distress_service = DistressService()


def llm_distress_score(text: str, conversationHistory: Optional[List[Any]] = None) -> float:
    """Legacy helper function for backwards compatibility."""
    return distress_service.score_distress_sync(text, conversationHistory)
