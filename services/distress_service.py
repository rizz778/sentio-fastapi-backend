"""
Distress Scoring Service
========================
Dedicated service for evaluating emotional distress scores using the side LLM.
Separated from session and turn orchestration.
"""

import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Optional
from google import genai
from google.genai import types
from core.config import settings
from prompts.distress import build_distress_prompt

logger = logging.getLogger(__name__)

VALID_EMOTIONS = {"anxious", "sad", "angry", "fear", "happy", "surprise", "disgust", "neutral"}


class DistressService:
    """
    Evaluates emotional distress scores (0.0 to 10.0) and primary text-based emotion
    from user input and recent conversation context.
    """

    def __init__(self):
        self._client: Optional[genai.Client] = None

    def _get_client(self) -> Optional[genai.Client]:
        if self._client is None:
            api_key = settings.DISTRESS_API_KEY or settings.GEMINI_API_KEY
            if not api_key:
                logger.warning("[DistressService] No API key configured for distress scoring.")
                return None
            api_version = getattr(settings, "GEMINI_API_VERSION", "v1beta") or "v1beta"
            self._client = genai.Client(
                api_key=api_key,
                http_options=types.HttpOptions(api_version=api_version),
            )
        return self._client

    def assess_distress_and_emotion_sync(
        self, text: str, conversation_history: Optional[List[Any]] = None
    ) -> Dict[str, Any]:
        """Synchronously invokes the side LLM to score distress and categorize emotion."""
        client = self._get_client()
        if not client:
            return {"distressScore": 0.0, "emotion": "neutral"}

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

            cleaned = re.sub(r"^```(?:json)?|```$", "", raw_text.strip(), flags=re.MULTILINE).strip()
            distress = 0.0
            emotion = "neutral"

            try:
                data = json.loads(cleaned)
                distress = float(data.get("distressScore", 0.0))
                em = str(data.get("emotion", "neutral")).lower().strip()
                if em in VALID_EMOTIONS:
                    emotion = em
                elif "anx" in em:
                    emotion = "anxious"
            except Exception:
                # Regex fallback for non-json response
                match = re.search(r"([0-9]+(?:\.[0-9]+)?)", raw_text)
                if match:
                    distress = float(match.group(1))
                for ve in VALID_EMOTIONS:
                    if ve in raw_text.lower():
                        emotion = ve
                        break

            clamped_distress = max(0.0, min(10.0, distress))
            logger.info(
                f"[DistressService] Assessed turn: distress={clamped_distress:.1f}, emotion={emotion}"
            )
            return {"distressScore": clamped_distress, "emotion": emotion}

        except Exception as e:
            logger.warning(f"[DistressService] Error assessing distress and emotion: {e}")
            return {"distressScore": 0.0, "emotion": "neutral"}

    def score_distress_sync(
        self, text: str, conversation_history: Optional[List[Any]] = None
    ) -> float:
        """Synchronously invokes the side LLM to score distress (float only)."""
        result = self.assess_distress_and_emotion_sync(text, conversation_history)
        return result.get("distressScore", 0.0)

    async def assess_turn(
        self, text: str, conversation_history: Optional[List[Any]] = None
    ) -> Dict[str, Any]:
        """Asynchronously evaluates distress score and emotion via thread pool."""
        return await asyncio.to_thread(
            self.assess_distress_and_emotion_sync, text, conversation_history
        )

    async def score_distress(
        self, text: str, conversation_history: Optional[List[Any]] = None
    ) -> float:
        """Asynchronously evaluates distress score via thread pool."""
        result = await self.assess_turn(text, conversation_history)
        return result.get("distressScore", 0.0)


distress_service = DistressService()


def llm_distress_score(text: str, conversationHistory: Optional[List[Any]] = None) -> float:
    """Legacy helper function for backwards compatibility."""
    return distress_service.score_distress_sync(text, conversationHistory)
