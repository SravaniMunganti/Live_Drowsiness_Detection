import cv2
import mediapipe as mp
import numpy as np
import winsound

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye_points):
    vertical_1 = np.linalg.norm(eye_points[1] - eye_points[5])  # Vertical line 1
    vertical_2 = np.linalg.norm(eye_points[2] - eye_points[4])  # Vertical line 2
    horizontal = np.linalg.norm(eye_points[0] - eye_points[3])  # Horizontal line
    return (vertical_1 + vertical_2) / (2.0 * horizontal)

# Eye landmarks indices for Mediapipe
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]

# Threshold and counters
EAR_THRESHOLD = 0.25  # EAR threshold to detect drowsiness
EAR_CONSEC_FRAMES = 20  # Number of consecutive frames to trigger drowsiness alert
COUNTER = 0  # Count of frames with EAR < threshold
ALARM_ON = False  # Alert status

# Function to play a beep sound
def play_alert_sound():
    frequency = 1000  # Set frequency of the sound
    duration = 1000   # Set duration in milliseconds (1 second)
    winsound.Beep(frequency, duration)

# Start webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Error: Cannot access the camera.")
        break

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    # Default status
    status_text = "AWAKE"

    # If face is detected
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get eye landmarks for both eyes
            left_eye = np.array([[face_landmarks.landmark[i].x * frame.shape[1],
                                  face_landmarks.landmark[i].y * frame.shape[0]] for i in LEFT_EYE_INDICES])
            right_eye = np.array([[face_landmarks.landmark[i].x * frame.shape[1],
                                   face_landmarks.landmark[i].y * frame.shape[0]] for i in RIGHT_EYE_INDICES])

            # Calculate EAR for both eyes
            left_EAR = eye_aspect_ratio(left_eye)
            right_EAR = eye_aspect_ratio(right_eye)
            avg_EAR = (left_EAR + right_EAR) / 2.0

            # Debug print: Show EAR values for both eyes and average EAR
            print(f"Left EAR: {left_EAR:.2f}, Right EAR: {right_EAR:.2f}, Average EAR: {avg_EAR:.2f}")

            # Check for drowsiness
            if avg_EAR < EAR_THRESHOLD:
                COUNTER += 1
                if COUNTER >= EAR_CONSEC_FRAMES:
                    ALARM_ON = True
                    status_text = "DROWSINESS DETECTED!"
                    # Play a beep sound when drowsiness is detected
                    play_alert_sound()
                    cv2.putText(frame, status_text, (100, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            else:
                COUNTER = 0
                ALARM_ON = False
                status_text = "AWAKE"

            # Display EAR for debugging
            cv2.putText(frame, f"EAR: {avg_EAR:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display status
    if ALARM_ON:
        cv2.putText(frame, "DROWSINESS DETECTED!", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    else:
        cv2.putText(frame, "AWAKE", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    # Show video feed
    cv2.imshow("Drowsiness Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
