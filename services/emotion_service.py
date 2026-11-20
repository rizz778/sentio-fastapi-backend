"""
Hosted Async Emotion Recognition Service
=========================================
Uses Gemini 2.5 Flash Vision asynchronously for non-blocking facial emotion analysis.
- 100% async network I/O (zero CPU/GIL blocking on the voice streaming event loop).
- Zero local ONNX/PyTorch dependencies or heavy binary overhead.
- In-flight request throttling to prevent network call pileups.
- Sliding-window smoothing to avoid emotion flicker.
"""

import asyncio
from collections import Counter, deque
import json
import logging
from typing import Deque, Dict, Optional, Set, Tuple

from google import genai
from google.genai import types

from core.config import settings

logger = logging.getLogger(__name__)

VALID_EMOTIONS = {"neutral", "happy", "sad", "angry", "fear", "surprise", "disgust"}

SYSTEM_PROMPT = """Analyze the facial expression of the person in the provided image frame.
Classify their primary emotional state into exactly one of: 'neutral', 'happy', 'sad', 'angry', 'fear', 'surprise', 'disgust'.
If no face or person is clearly visible, return 'neutral'.
Return ONLY valid JSON matching this schema:
{"emotion": "<one_of_the_seven_emotions>", "confidence": 0.0_to_1.0}"""


class HostedEmotionService:
    def __init__(self, buffer_size: int = 5):
        self.buffer_size = buffer_size
        self._client: Optional[genai.Client] = None
        self._session_buffers: Dict[str, Deque[str]] = {}
        # Track active in-flight requests per session to drop intermediate frames
        self._in_flight_sessions: Set[str] = set()

    def _get_client(self) -> Optional[genai.Client]:
        if self._client is None:
            if not settings.GEMINI_API_KEY:
                logger.warning("[EmotionService] GEMINI_API_KEY not configured.")
                return None
            self._client = genai.Client(api_key=settings.GEMINI_API_KEY)
        return self._client

    def _get_buffer(self, session_id: str) -> Deque[str]:
        if session_id not in self._session_buffers:
            self._session_buffers[session_id] = deque(maxlen=self.buffer_size)
        return self._session_buffers[session_id]

    def cleanup_session(self, session_id: str) -> None:
        self._session_buffers.pop(session_id, None)
        self._in_flight_sessions.discard(session_id)

    async def analyze_frame_bytes_async(
        self,
        image_bytes: bytes,
        mime_type: str = "image/jpeg",
        session_id: Optional[str] = None,
    ) -> Tuple[str, str]:
        """
        Asynchronously analyzes an image frame using Gemini Flash Vision.
        Returns: (current_emotion, aggregated_smoothed_emotion)
        """
        if not image_bytes:
            return "neutral", "neutral"

        active_sid = session_id or "default"

        # Throttle: drop intermediate frame if previous frame request is still in-flight
        if active_sid in self._in_flight_sessions:
            buf = self._get_buffer(active_sid)
            last = buf[-1] if buf else "neutral"
            return last, last

        client = self._get_client()
        if not client:
            return "neutral", "neutral"

        self._in_flight_sessions.add(active_sid)
        try:
            image_part = types.Part.from_bytes(
                data=image_bytes,
                mime_type=mime_type or "image/jpeg",
            )

            # 100% asynchronous network call — does not block asyncio loop
            response = await client.aio.models.generate_content(
                model=settings.GEMINI_VISION_MODEL,
                contents=[image_part, SYSTEM_PROMPT],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.1,
                ),
            )

            raw_text = (response.text or "").strip()
            parsed = json.loads(raw_text)
            detected = str(parsed.get("emotion", "neutral")).lower().strip()
            if detected not in VALID_EMOTIONS:
                detected = "neutral"

            # Update temporal smoothing buffer
            buf = self._get_buffer(active_sid)
            buf.append(detected)

            most_common = Counter(buf).most_common(1)
            aggregated = most_common[0][0] if most_common else detected

            logger.debug(
                f"[EmotionService] Frame analyzed: {detected} (smooth: {aggregated}) for {active_sid}"
            )
            return detected, aggregated

        except Exception as e:
            logger.warning(f"[EmotionService] Hosted vision analysis error: {e}")
            buf = self._get_buffer(active_sid)
            fallback = buf[-1] if buf else "neutral"
            return fallback, fallback
        finally:
            self._in_flight_sessions.discard(active_sid)


# Module-level singleton
emotion_service = HostedEmotionService()
EmotionService = HostedEmotionService
