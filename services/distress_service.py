import logging
import re
from typing import Any, List, Optional
from google import genai
from core.config import settings
from prompts.distress import build_distress_prompt

logger = logging.getLogger(__name__)


def llm_distress_score(text: str, conversationHistory: Optional[List[Any]] = None) -> float:
    if not settings.GEMINI_API_KEY:
        logger.warning("[DistressService] GEMINI_API_KEY not configured. Returning 0.0.")
        return 0.0

    prompt = build_distress_prompt(text, conversationHistory)

    try:
        client = genai.Client(api_key=settings.GEMINI_API_KEY)
        response = client.models.generate_content(
            model=settings.GEMINI_DISTRESS_MODEL,
            contents=prompt,
        )
        raw_text = ""
        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            text_parts = [p.text for p in response.candidates[0].content.parts if getattr(p, "text", None)]
            raw_text = "".join(text_parts).strip()
        if not raw_text:
            raw_text = (getattr(response, "text", None) or "").strip()

        # Extract the first number between 0 and 10
        match = re.search(r"([0-9]+(?:\.[0-9]+)?)", raw_text)
        if match:
            score = float(match.group(1))
            return max(0.0, min(10.0, score))
        logger.warning(f"[DistressService] Could not parse numeric score from output: '{raw_text}'")
        return 0.0
    except Exception as e:
        logger.warning(f"[DistressService] Error assessing distress score: {e}")
        return 0.0