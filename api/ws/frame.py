import asyncio
import base64
import json
import logging
import time
from typing import Optional
import cv2
from fastapi import APIRouter, Depends, Query, WebSocket
import numpy as np

from api.deps import get_emotion_service
from api.ws.state import cleanup_client, get_or_create_buffer, set_client_emotion
from services.emotion_service import EmotionService

logger = logging.getLogger(__name__)

router = APIRouter()


@router.websocket("/frame")
async def frame_websocket_endpoint(
    websocket: WebSocket,
    client_id: Optional[str] = Query(None),
    emotion_service: EmotionService = Depends(get_emotion_service),
):
    await websocket.accept()
    resolved_client_id = client_id or f"client_{id(websocket)}"
    buffer = get_or_create_buffer(resolved_client_id)
    logger.info(f"[FrameWS] Accepted frame connection for {resolved_client_id}")

    try:
        while True:
            msg = await websocket.receive_text()
            try:
                data = json.loads(msg)
            except json.JSONDecodeError:
                await websocket.send_json({"error": "Invalid JSON"})
                continue

            if not isinstance(data, dict):
                await websocket.send_json({"error": "Message must be a JSON object"})
                continue

            if "frame" in data:
                try:
                    frame_b64 = data["frame"]
                    img_bytes = base64.b64decode(frame_b64)
                    np_arr = np.frombuffer(img_bytes, np.uint8)
                    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                    if frame is None:
                        logger.error(f"[FrameWS] Failed to decode frame for {resolved_client_id}")
                        await websocket.send_json({"error": "Invalid frame"})
                        continue

                    if frame.mean() < 5.0:
                        logger.warning(f"[FrameWS] Black frame received for {resolved_client_id}")
                        await websocket.send_json({"error": "Black frame received"})
                        continue

                    current_emotion = await asyncio.to_thread(
                        emotion_service.analyze_frame, frame, buffer
                    )
                    aggregated = emotion_service.get_aggregated_emotion(buffer=buffer)

                    emotion_payload = {
                        "current": current_emotion,
                        "aggregated": aggregated,
                        "timestamp": time.time(),
                    }

                    set_client_emotion(resolved_client_id, emotion_payload)

                    await websocket.send_json({
                        "emotion": {
                            "current": current_emotion,
                            "aggregated": aggregated,
                        }
                    })
                except Exception as e:
                    logger.exception(f"[FrameWS] Error processing frame: {e}")
                    await websocket.send_json({"error": "Emotion processing failed"})
            elif data.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
            else:
                await websocket.send_json({"error": "Unsupported message format, expected 'frame'"})

    except Exception as e:
        logger.error(f"[FrameWS] Disconnected {resolved_client_id}: {e}")
    finally:
        cleanup_client(resolved_client_id)
        logger.info(f"[FrameWS] Cleaned up frame session for {resolved_client_id}")
