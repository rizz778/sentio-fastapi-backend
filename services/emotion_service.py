from collections import Counter, deque
import logging
from typing import Deque, Optional
import cv2
import numpy as np

logger = logging.getLogger(__name__)

_recognizer = None
_face_cascade = None
_recognizer_load_error = None

# Normalized mappings to maintain compatibility across the system
EMOTION_MAP = {
    "anger": "angry",
    "contempt": "neutral",
    "disgust": "disgust",
    "fear": "fear",
    "happiness": "happy",
    "neutral": "neutral",
    "sadness": "sad",
    "surprise": "surprise",
}


def get_emotion_engine():
    """Lazy-load the EmotiEffLib ONNX recognizer and OpenCV face cascade."""
    global _recognizer, _face_cascade, _recognizer_load_error
    if _recognizer_load_error is not None:
        raise _recognizer_load_error
    if _recognizer is None:
        try:
            from emotiefflib.facial_analysis import EmotiEffLibRecognizer

            face_cascade = None
            if hasattr(cv2, "CascadeClassifier") and hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
                cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                try:
                    cascade = cv2.CascadeClassifier(cascade_path)
                    if not cascade.empty():
                        face_cascade = cascade
                except Exception as cascade_err:
                    logger.warning(f"[EmotionService] Face cascade load failed, falling back to full-frame: {cascade_err}")

            recognizer = EmotiEffLibRecognizer(
                model_name="enet_b0_8_best_vgaf", engine="onnx"
            )

            _face_cascade = face_cascade
            _recognizer = recognizer
            logger.info("[EmotionService] Successfully initialized EmotiEffLib ONNX model")
        except Exception as exc:
            _recognizer_load_error = exc
            logger.error(f"[EmotionService] Failed to initialize emotion engine: {exc}")
            raise
    return _recognizer, _face_cascade


class EmotionService:
    def __init__(self, buffer_size: int = 10):
        self.buffer: Deque[str] = deque(maxlen=buffer_size)

    def analyze_frame(self, frame: np.ndarray, buffer: Optional[Deque[str]] = None) -> str:
        try:
            if frame is None or frame.size == 0:
                return "unknown"

            recognizer, face_cascade = get_emotion_engine()

            # Handle color channel format conversion
            if len(frame.shape) == 2:
                gray = frame
                rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            elif frame.shape[2] == 4:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
            else:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces with Haar cascade if available
            faces = []
            if face_cascade is not None:
                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(48, 48),
                )

            if len(faces) > 0:
                # Select the largest face by area (w * h)
                x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
                # Add slight margin around the face if possible
                pad_x = int(0.1 * w)
                pad_y = int(0.1 * h)
                x1 = max(0, x - pad_x)
                y1 = max(0, y - pad_y)
                x2 = min(rgb.shape[1], x + w + pad_x)
                y2 = min(rgb.shape[0], y + h + pad_y)
                face_img = rgb[y1:y2, x1:x2]
            else:
                # Fallback to full frame
                face_img = rgb

            predicted_emotions, _ = recognizer.predict_emotions(face_img, logits=False)
            raw_emotion = (
                predicted_emotions[0].lower() if predicted_emotions else "neutral"
            )
            emotion = EMOTION_MAP.get(raw_emotion, raw_emotion)

            logger.debug(f"[Emotion] EmotiEffLib raw: {raw_emotion} -> {emotion}")

            target_buffer = buffer if buffer is not None else self.buffer
            target_buffer.append(emotion)
            return emotion
        except Exception as e:
            logger.warning(f"Emotion analysis failed: {e}")
            return "unknown"

    def get_aggregated_emotion(self, buffer: Optional[Deque[str]] = None) -> str:
        buf = buffer if buffer is not None else self.buffer
        if not buf:
            return "neutral"
        most_common = Counter(buf).most_common(1)
        return most_common[0][0]
