import logging
from typing import Optional
from core.config import settings
from services.emotion_service import EmotionService
from services.gemini_service import GeminiService

logger = logging.getLogger(__name__)


class DependencyStorage:
    def __init__(self, api_key: Optional[str], gemini_model: str):
        self.api_key = api_key
        self.gemini_model = gemini_model
        if api_key:
            self._gemini_service: Optional[GeminiService] = GeminiService(
                api_key=api_key, model=gemini_model
            )
        else:
            logger.warning("[Deps] GEMINI_API_KEY is not set. GeminiService will not be initialized.")
            self._gemini_service = None
        self._emotion_service = EmotionService()

    def get_gemini_service(self) -> Optional[GeminiService]:
        return self._gemini_service

    def get_emotion_service(self) -> EmotionService:
        return self._emotion_service


dependency_storage: Optional[DependencyStorage] = None


async def initialize_dependencies():
    global dependency_storage
    dependency_storage = DependencyStorage(
        api_key=settings.GEMINI_API_KEY,
        gemini_model=settings.GEMINI_TEXT_MODEL or settings.GEMINI_MODEL,
    )


def get_gemini_service() -> GeminiService:
    if dependency_storage is None:
        raise RuntimeError("Dependencies not initialized")
    service = dependency_storage.get_gemini_service()
    if service is None:
        raise RuntimeError("GEMINI_API_KEY is not configured in environment or .env")
    return service


def get_emotion_service() -> EmotionService:
    if dependency_storage is None:
        raise RuntimeError("Dependencies not initialized")
    return dependency_storage.get_emotion_service()
