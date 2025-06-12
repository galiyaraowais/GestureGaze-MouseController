import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import keyboard
import time

# Initialize Mediapipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Set up webcam (increase FPS & visibility)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # For Windows
cap.set(cv2.CAP_PROP_FPS, 30)  # Increase FPS for smoother tracking
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Get screen size
screen_width, screen_height = pyautogui.size()

# Adjust EAR (Eye Aspect Ratio) threshold dynamically
BLINK_THRESHOLD = 0.18  # Lower value for better detection

# Get eye aspect ratio (EAR)
def eye_aspect_ratio(eye_landmarks):
    p1, p2, p3, p4, p5, p6 = eye_landmarks
    vertical_1 = np.linalg.norm(np.array(p2) - np.array(p6))
    vertical_2 = np.linalg.norm(np.array(p3) - np.array(p5))
    horizontal = np.linalg.norm(np.array(p1) - np.array(p4))
    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear

# Detect blink and wink
def detect_blink_wink(landmarks):
    left_eye_indices = [33, 160, 158, 133, 153, 144]  # Left eye
    right_eye_indices = [362, 385, 387, 263, 373, 380]  # Right eye

    left_eye = [landmarks[i] for i in left_eye_indices]
    right_eye = [landmarks[i] for i in right_eye_indices]

    left_EAR = eye_aspect_ratio(left_eye)
    right_EAR = eye_aspect_ratio(right_eye)

    if left_EAR < BLINK_THRESHOLD and right_EAR < BLINK_THRESHOLD:
        return "Blink"
    elif left_EAR < BLINK_THRESHOLD:
        return "Wink Left"
    elif right_EAR < BLINK_THRESHOLD:
        return "Wink Right"
    return None

# Move the cursor based on eye position
def move_cursor(x, y):
    new_x = np.interp(x, [0, 640], [0, screen_width])
    new_y = np.interp(y, [0, 480], [0, screen_height])
    pyautogui.moveTo(new_x, new_y, duration=0.05)  # Faster response

# Start tracking
print("\n🔍 Press 'q' to stop tracking...")
frame_count = 0
blink_time = 0  # Prevent multiple clicks for one blink

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror effect for natural tracking
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = [(int(l.x * 640), int(l.y * 480)) for l in face_landmarks.landmark]

            eye_x, eye_y = landmarks[168]  # Eye center landmark
            move_cursor(eye_x, eye_y)

            action = detect_blink_wink(landmarks)
            if action == "Blink":
                if time.time() - blink_time > 0.5:  # Prevent double clicks
                    pyautogui.click()
                    print("🖱️ Left Click!")
                    blink_time = time.time()
            elif action == "Wink Left" or action == "Wink Right":
                pyautogui.rightClick()
                print("🖱️ Right Click!")

    # Show webcam feed
    cv2.imshow("Eye Mouse Control", frame)
    frame_count += 1

    if keyboard.is_pressed('q'):
        print("\n👋 Stopping gaze tracking...")
        break

cap.release()
cv2.destroyAllWindows()
