import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
import keyboard
from sklearn.ensemble import RandomForestClassifier
import ast
import pandas as pd

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Disable PyAutoGUI fail-safe
pyautogui.FAILSAFE = False

# Load the dataset
df = pd.read_csv("classified_gaze.csv")
print(df.head())  # Print the first few rows of the dataset

# Extract features and labels
X = df["look_vec"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
y = df["gaze_direction"]

# Convert feature vectors into a NumPy array
X = np.array(X.tolist())

# Train the model
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X, y)
print("\n✅ Model trained with accuracy:", clf.score(X, y))

# Function to simulate gaze tracking and move the mouse
def move_cursor(predicted_direction):
    screen_width, screen_height = pyautogui.size()
    current_x, current_y = pyautogui.position()
    step = 70  # Increased movement speed
    
    if predicted_direction == "Right":
        pyautogui.moveTo(min(current_x + step, screen_width), current_y, duration=0.05)
    elif predicted_direction == "Left":
        pyautogui.moveTo(max(current_x - step, 0), current_y, duration=0.05)
    elif predicted_direction == "Up":
        pyautogui.scroll(10)  # Scroll up
    elif predicted_direction == "Down":
        pyautogui.scroll(-10)  # Scroll down
    elif predicted_direction == "Blink":
        pyautogui.click()
        print("🖱️ Left Click!")
    elif predicted_direction == "Wink":
        pyautogui.rightClick()
        print("🖱️ Right Click!")
    elif predicted_direction == "Squint":
        pyautogui.mouseDown()
        print("🔁 Holding Drag!")
    else:
        pyautogui.mouseUp()

# Open webcam feed
cap = cv2.VideoCapture(0)
print("\n🔍 Press 'q' to stop tracking...")

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip frame for natural interaction
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        # Display the webcam feed
        cv2.imshow("Eye Tracking", frame)

        # Simulate a random gaze vector (Replace this with real-time gaze data in future)
        random_gaze_vector = np.random.rand(1, 4)  # Assuming 4D feature vector
        predicted_direction = clf.predict(random_gaze_vector)[0]
        print("Predicted gaze direction:", predicted_direction)
        move_cursor(predicted_direction)
        
        # Check for 'q' key press to exit
        if keyboard.is_pressed('q'):
            print("\n👋 Stopping gaze tracking...")
            break
        
        time.sleep(0.5)  # Reduced delay for faster response

except KeyboardInterrupt:
    print("\n👋 Stopping gaze tracking...")
finally:
    cap.release()
    cv2.destroyAllWindows()
