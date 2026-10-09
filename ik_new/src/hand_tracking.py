
# Hand tracking using YOLOv8 community trained model and OpenCV. Hand keypoints by Mediapipe

import cv2
import mediapipe as mp
from ultralytics import YOLO
import torch

video_path = "media/abacus.mp4"

# Community-trained hand-detection weights (auto-downloaded on first run).
HAND_MODEL = "https://huggingface.co/Bingsu/adetailer/resolve/main/hand_yolov8n.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", DEVICE)

model = YOLO(HAND_MODEL)
model.to(DEVICE)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise IOError(f"Cannot open video file {video_path}")

cv2.namedWindow("Raw video", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Raw video", 480, 640)

cv2.namedWindow("hand tracking", cv2.WINDOW_NORMAL)
cv2.resizeWindow("hand tracking", 480, 640)

cv2.namedWindow("mediapipe keypoints", cv2.WINDOW_NORMAL)
cv2.resizeWindow("mediapipe keypoints", 480, 640)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    cv2.imshow("Raw video", frame)

    results = model.track(frame, persist=True, verbose=False, device=DEVICE)
    tracked_frame = results[0].plot()
    cv2.imshow("hand tracking", tracked_frame)

    keypoints_frame = frame.copy()
    mediapipe_results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if mediapipe_results.multi_hand_landmarks:
        for hand_landmarks in mediapipe_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                keypoints_frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )
    cv2.imshow("mediapipe keypoints", keypoints_frame)

    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

hands.close()
cap.release()
cv2.destroyAllWindows()
