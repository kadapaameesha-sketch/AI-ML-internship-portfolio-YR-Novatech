import cv2
import mediapipe as mp
import pyautogui

# Initialize MediaPipe Hands and OpenCV Video Capture
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# Get screen width and height for coordinate mapping
screen_width, screen_height = pyautogui.size()

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Flip the image horizontally for a selfie-view display and convert BGR to RGB
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image and find hands
    results = hands.process(image_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Get landmark coordinates for Index Finger Tip (ID 8) and Thumb Tip (ID 4)
            h, w, c = image.shape
            index_finger = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_finger = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            
            x_index, y_index = int(index_finger.x * w), int(index_finger.y * h)
            x_thumb, y_thumb = int(thumb_finger.x * w), int(thumb_finger.y * h)
            
            # Map webcam coordinates to screen resolution for cursor movement
            screen_x = int(index_finger.x * screen_width)
            screen_y = int(index_finger.y * screen_height)
            
            # Move mouse cursor
            pyautogui.moveTo(screen_x, screen_y)
            
            # Calculate distance between thumb and index finger for clicking action
            distance = ((x_thumb - x_index)**2 + (y_thumb - y_index)**2) ** 0.5
            if distance < 35:
                pyautogui.click()
                pyautogui.sleep(0.2) # Prevent multiple rapid clicks

    # Display the webcam window
    cv2.imshow('AI Hand Gesture Computer Control', image)
    
    if cv2.waitKey(5) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
