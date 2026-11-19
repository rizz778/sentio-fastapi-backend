import asyncio
import logging
from typing import Optional
import grpc
from proto import sentio_gateway_pb2_grpc as pb2_grpc
from grpc_server.servicer import SentioLiveGatewayServicer
from core.config import settings

logger = logging.getLogger(__name__)

_grpc_server: Optional[grpc.aio.Server] = None


async def start_grpc_server(host: Optional[str] = None, port: Optional[int] = None) -> grpc.aio.Server:
    global _grpc_server
    host = host or settings.GRPC_HOST
    port = port or settings.GRPC_PORT
    bind_address = f"{host}:{port}"

    server = grpc.aio.server(
        options=[
            ("grpc.max_receive_message_length", 16 * 1024 * 1024),
            ("grpc.max_send_message_length", 16 * 1024 * 1024),
            ("grpc.keepalive_time_ms", 10000),
            ("grpc.keepalive_timeout_ms", 5000),
            ("grpc.keepalive_permit_without_calls", 1),
            ("grpc.http2.max_pings_without_data", 0),
        ]
    )

    pb2_grpc.add_SentioLiveGatewayServicer_to_server(
        SentioLiveGatewayServicer(), server
    )

    server.add_insecure_port(bind_address)
    logger.info(f"[gRPC] Starting SentioLiveGateway gRPC server on {bind_address}")
    await server.start()
    _grpc_server = server
    return server


async def stop_grpc_server(grace: float = 5.0) -> None:
    global _grpc_server
    if _grpc_server is not None:
        logger.info(f"[gRPC] Stopping SentioLiveGateway gRPC server (grace={grace}s)...")
        await _grpc_server.stop(grace)
        _grpc_server = None
        logger.info("[gRPC] SentioLiveGateway gRPC server stopped.")
