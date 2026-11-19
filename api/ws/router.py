from fastapi import APIRouter
from api.ws.audio import router as audio_router
from api.ws.frame import router as frame_router
from api.ws.text import router as text_router

router = APIRouter(tags=["ws"])

# Distinct WebSocket endpoints
router.include_router(text_router)
router.include_router(frame_router)
router.include_router(audio_router)

__all__ = ["router"]
