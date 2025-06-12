import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np
import os

# Initialize camera and Mediapipe Hand Tracking
cam = cv2.VideoCapture(0)
hands = mp.solutions.hands.Hands(min_detection_confidence=0.90, min_tracking_confidence=0.90)
drawing_utils = mp.solutions.drawing_utils
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence=0.95, min_tracking_confidence=0.95)
screen_w, screen_h = pyautogui.size()

# Track frames where two hands are detected
two_hand_frames = 0  
TERMINATION_THRESHOLD = 5  # Number of frames with 2 hands before terminating

# Function to detect hand gestures
def detect_hand_gesture(landmarks, hand_count):
    global two_hand_frames
    thumb_up = landmarks[4].y < landmarks[3].y and landmarks[4].x < landmarks[3].x
    thumb_down = landmarks[4].y > landmarks[3].y and landmarks[4].x > landmarks[3].x
    index_finger = landmarks[8].y < landmarks[6].y
    little_finger = landmarks[20].y < landmarks[18].y
    two_finger_swipe_up = landmarks[8].y < landmarks[7].y and landmarks[12].y < landmarks[11].y
    two_finger_swipe_down = landmarks[8].y > landmarks[7].y and landmarks[12].y > landmarks[11].y
    
    if hand_count > 1:
        two_hand_frames += 1
        if two_hand_frames >= TERMINATION_THRESHOLD:
            return "Terminate"
    else:
        two_hand_frames = 0  # Reset if only one hand is detected
    
    if index_finger:
        return "Left Click"
    elif little_finger:
        return "Right Click"
    elif thumb_up:
        return "Left Drag"
    elif thumb_down:
        return "Right Drag"
    elif two_finger_swipe_up:
        return "Scroll Up"
    elif two_finger_swipe_down:
        return "Scroll Down"
    return None

# Function to detect eye gestures
def detect_eye_gesture(landmarks, frame):
    left_eye = [landmarks[145], landmarks[159]]
    right_eye = [landmarks[374], landmarks[386]]
    left_ear = abs(left_eye[0].y - left_eye[1].y)
    right_ear = abs(right_eye[0].y - right_eye[1].y)
    
    # Add visualization elements to eyes
    for landmark in left_eye + right_eye:
        x = int(landmark.x * frame.shape[1])
        y = int(landmark.y * frame.shape[0])
        cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
    
    if left_ear < 0.003 and right_ear < 0.003:
        os.system('explorer')  # Open File Explorer
        return "Blink"
    elif left_ear < 0.003:
        pyautogui.hotkey('win', 'down')  # Minimize window
        return "Wink"
    return None

# Main loop
while True:
    _, frame = cam.read()
    if not _:
        print("Error: Unable to capture video frame.")
        continue
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_h, frame_w, _ = frame.shape
    
    # Hand tracking
    output = hands.process(rgb_frame)
    hand_count = 0
    if output.multi_hand_landmarks:
        hand_count = len(output.multi_hand_landmarks)
        for hand_landmarks in output.multi_hand_landmarks:
            drawing_utils.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
            landmarks = hand_landmarks.landmark
            cursor_x = np.interp(landmarks[8].x, [0, 1], [0, screen_w])
            cursor_y = np.interp(landmarks[8].y, [0, 1], [0, screen_h])
            pyautogui.moveTo(cursor_x, cursor_y, duration=0.03)  # Faster response time
            
            action = detect_hand_gesture(landmarks, hand_count)
            if action == "Left Click":
                pyautogui.click()
            elif action == "Right Click":
                pyautogui.rightClick()
            elif action == "Left Drag":
                pyautogui.mouseDown()
            elif action == "Right Drag":
                pyautogui.mouseDown(button='right')
            elif action == "Scroll Up":
                pyautogui.scroll(10)
            elif action == "Scroll Down":
                pyautogui.scroll(-10)
            elif action == "Terminate":
                print("👋 Terminating mouse control...")
                cam.release()
                cv2.destroyAllWindows()
                exit()
            else:
                pyautogui.mouseUp()
    
    # Eye tracking
    output = face_mesh.process(rgb_frame)
    if output.multi_face_landmarks:
        try:
            landmarks = output.multi_face_landmarks[0].landmark
            detect_eye_gesture(landmarks, frame)
        except Exception as e:
            print("Error processing face landmarks:", e)
    
    # Ensure webcam preview remains visible
    cv2.imshow('Hand & Eye Controlled Mouse', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("👋 Stopping gaze tracking...")
        break

cam.release()
cv2.destroyAllWindows()
