from contextlib import asynccontextmanager
import uuid
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.middleware.trustedhost import TrustedHostMiddleware
import structlog

from core.config import settings
from core.exceptions import AppException
from core.handlers import (
    app_exception_handler,
    generic_exception_handler,
    validation_exception_handler,
)
from core.json import custom_jsonable_encoder
from core.logging import setup_logging
from grpc_server import start_grpc_server, stop_grpc_server


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()

    # Start SentioLiveGateway gRPC server
    try:
        await start_grpc_server()
    except Exception as e:
        structlog.get_logger(__name__).error("Failed to start gRPC server", error=str(e))

    yield

    # Graceful shutdown of gRPC server
    await stop_grpc_server(grace=3.0)


class CustomJSONResponse(JSONResponse):
    def render(self, content) -> bytes:
        return super().render(custom_jsonable_encoder(content))


app = FastAPI(
    title="Sentio AI Engine Backend",
    description="Multimodal Emotion & Voice AI Backend for Sentio",
    version="1.0.0",
    lifespan=lifespan,
    default_response_class=CustomJSONResponse,
)

# Exception Handlers
app.add_exception_handler(AppException, app_exception_handler)
app.add_exception_handler(ValidationError, validation_exception_handler)
app.add_exception_handler(Exception, generic_exception_handler)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    # Prefer correlation ID from gateway if provided
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    request.state.request_id = request_id
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(request_id=request_id)
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status": "healthy",
        "service": "sentio-ai-engine",
        "grpc_port": settings.GRPC_PORT,
    }


@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Sentio AI Engine Backend is active",
        "grpc": f"{settings.GRPC_HOST}:{settings.GRPC_PORT}",
    }
