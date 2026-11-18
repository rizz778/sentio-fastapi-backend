from fastapi import APIRouter
from api.ws.audio import router as audio_router
from api.ws.frame import router as frame_router, frame_websocket_endpoint
from api.ws.text import router as text_router

router = APIRouter(tags=["ws"])

# Distinct WebSocket endpoints
router.include_router(text_router)
router.include_router(frame_router)
router.include_router(audio_router)

# Fallback alias: allow connections to root '/' to route to emotion frame stream
router.add_api_websocket_route("/", frame_websocket_endpoint)

__all__ = ["router"]
