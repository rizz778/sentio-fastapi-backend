import sys
import os

# Add server root to sys.path if not present
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.emotion_service import EmotionService

def main():
    emotion_service = EmotionService()
    cap = cv2.VideoCapture(0)
    print("Press SPACE to capture a frame and analyze emotion, or ESC to exit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
        cv2.imshow("Webcam", frame)
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break
        elif key == 32:  # SPACE
            emotion = emotion_service.analyze_frame(frame)
            print(f"Detected emotion: {emotion}")
            print(f"Buffer: {list(emotion_service.buffer)}")
            print(f"Aggregated emotion: {emotion_service.get_aggregated_emotion()}")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()