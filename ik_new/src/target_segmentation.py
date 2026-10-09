
# Hand keypoints Mediapipe and segmentation of targets using SAM
import cv2
import numpy as np
import mediapipe as mp
import torch
from ultralytics.models.sam import SAM2VideoPredictor

video_path = "media/abacus.mp4"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", DEVICE)

sam_overrides = dict(
    conf=0.25,
    task="segment",
    mode="predict",
    imgsz=1024,
    model="sam2.1_b.pt",
    device=DEVICE,
)
sam_predictor = SAM2VideoPredictor(overrides=sam_overrides)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise IOError(f"Cannot open video file {video_path}")

# --- Preview the first frame and let the user click a target point ---
ret, raw_first_frame = cap.read()
cap.release()
if not ret:
    raise IOError("Cannot read first frame from video")

first_frame = cv2.rotate(raw_first_frame, cv2.ROTATE_90_CLOCKWISE)
rotated_width = first_frame.shape[1]

TARGET_COLORS = [
    (0, 255, 0),
    (0, 165, 255),
    (255, 0, 0),
    (0, 255, 255),
    (255, 0, 255),
]

click_points = []
preview_window = "Select targets - click points, press any key to continue"
cv2.namedWindow(preview_window, cv2.WINDOW_NORMAL)
cv2.resizeWindow(preview_window, 480, 640)


def on_mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        click_points.append((x, y))
        preview = first_frame.copy()
        for i, pt in enumerate(click_points):
            color = TARGET_COLORS[i % len(TARGET_COLORS)]
            cv2.drawMarker(preview, pt, color, cv2.MARKER_CROSS, 20, 2)
        cv2.imshow(preview_window, preview)


cv2.setMouseCallback(preview_window, on_mouse_click)
cv2.imshow(preview_window, first_frame)

while True:
    key = cv2.waitKey(20) & 0xFF
    if key == 255:
        continue
    if not click_points:
        if key == ord("q"):
            break
        continue
    break
cv2.destroyWindow(preview_window)

if not click_points:
    raise RuntimeError("No target points selected")

print("Selected points (rotated frame):", click_points)

# The predictor decodes the raw (unrotated) video, so the click points picked on
# the rotated preview must be mapped back to raw-frame coordinates.
sam_points = [[[pt[1], rotated_width - 1 - pt[0]]] for pt in click_points]
sam_labels = [[1] for _ in click_points]

# Propagate each point prompt (a separate tracked object) from the first frame
# across the whole video.
sam_results = sam_predictor(source=video_path, points=sam_points, labels=sam_labels)

cv2.namedWindow("Raw video", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Raw video", 480, 640)

cv2.namedWindow("mediapipe keypoints", cv2.WINDOW_NORMAL)
cv2.resizeWindow("mediapipe keypoints", 480, 640)

for result in sam_results:
    frame = cv2.rotate(result.orig_img, cv2.ROTATE_90_CLOCKWISE)

    cv2.imshow("Raw video", frame)

    # Mediapipe hand keypoints
    keypoints_frame = frame.copy()

    # Overlay each propagated SAM target segmentation mask
    if result.masks is not None:
        for i, raw_mask in enumerate(result.masks.data.cpu().numpy()):
            color = np.array(TARGET_COLORS[i % len(TARGET_COLORS)])
            target_mask = cv2.rotate(raw_mask.astype("uint8"), cv2.ROTATE_90_CLOCKWISE).astype(bool)
            keypoints_frame[target_mask] = (
                0.5 * color + 0.5 * keypoints_frame[target_mask]
            ).astype("uint8")
        for i, pt in enumerate(click_points):
            color = TARGET_COLORS[i % len(TARGET_COLORS)]
            cv2.drawMarker(keypoints_frame, pt, color, cv2.MARKER_CROSS, 20, 2)

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
cv2.destroyAllWindows()
