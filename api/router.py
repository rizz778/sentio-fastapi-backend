from fastapi import APIRouter
from api.routes.distress_score import router as distress_router
from api.routes.health import router as health_router
from api.ws.router import router as ws_router

api_router = APIRouter()

# HTTP routes
api_router.include_router(health_router)
api_router.include_router(distress_router)

# WebSocket routes
api_router.include_router(ws_router)

__all__ = ["api_router"]
