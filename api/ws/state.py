from collections import deque
from typing import Any, Deque, Dict, Optional

# Stores per-client emotion state: client_id -> {"current": str, "aggregated": str, "timestamp": float}
client_emotions: Dict[str, Dict[str, Any]] = {}

# Stores per-client frame history buffer for emotion aggregation
client_buffers: Dict[str, Deque[str]] = {}

# Fallback latest detected emotion across connections
latest_global_emotion: Dict[str, Any] = {}


def set_client_emotion(client_id: str, emotion_data: Dict[str, Any]) -> None:
    client_emotions[client_id] = emotion_data
    latest_global_emotion.clear()
    latest_global_emotion.update(emotion_data)


def get_client_emotion(client_id: Optional[str] = None) -> Optional[str]:
    if client_id and client_id in client_emotions:
        data = client_emotions[client_id]
        if isinstance(data, dict):
            return data.get("current") or data.get("aggregated")
        return str(data)
    if latest_global_emotion:
        return latest_global_emotion.get("current") or latest_global_emotion.get("aggregated")
    return None


def get_or_create_buffer(client_id: str, maxlen: int = 10) -> Deque[str]:
    if client_id not in client_buffers:
        client_buffers[client_id] = deque(maxlen=maxlen)
    return client_buffers[client_id]


def cleanup_client(client_id: str) -> None:
    client_emotions.pop(client_id, None)
    client_buffers.pop(client_id, None)
