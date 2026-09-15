import cv2
import mediapipe as mp

# Initialize MediaPipe Hands utilities and model
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

# Open the default webcam
cap = cv2.VideoCapture(0)

print("Starting AI Hand Gesture Recognition... Press 'q' to quit.")

while cap.isOpened():
  success, frame = cap.read()
  if not success:
    print("Ignoring empty camera frame from webcam.")
    continue

  # Mirror the frame horizontally for natural interaction
  frame = cv2.flip(frame, 1)
  h, w, c = frame.shape

  # Convert the BGR frame to RGB for MediaPipe processing
  image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  image.flags.writeable = False
  results = hands.process(image)

  # Convert back to BGR for OpenCV rendering
  image.flags.writeable = True
  image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

  gesture_text = "Detecting..."

  if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
      # Draw hand skeleton connections on the frame
      mp_drawing.draw_landmarks(
          image, hand_landmarks, mp_hands.HAND_CONNECTIONS
      )

      # Extract landmark coordinates
      lm = hand_landmarks.landmark

      # Finger openness evaluation based on landmark y-coordinates
      index_open = lm[8].y < lm[6].y
      middle_open = lm[12].y < lm[10].y
      ring_open = lm[16].y < lm[14].y
      pinky_open = lm[20].y < lm[18].y

      # Rule-based gesture classification logic
      if index_open and middle_open and ring_open and pinky_open:
        gesture_text = "Open Hand"
      elif (
          not index_open
          and not middle_open
          and not ring_open
          and not pinky_open
      ):
        gesture_text = "Fist"
      elif index_open and middle_open and not ring_open and not pinky_open:
        gesture_text = "Victory / Peace"
      elif lm[4].y < lm[8].y and not index_open and not middle_open:
        gesture_text = "Thumbs Up"
      else:
        gesture_text = "Unknown Gesture"

  # Overlay the recognized gesture text onto the live video feed
  cv2.putText(
      image,
      f"Gesture: {gesture_text}",
      (30, 50),
      cv2.FONT_HERSHEY_SIMPLEX,
      1,
      (0, 255, 0),
      2,
      cv2.LINE_AA,
  )

  # Display the live window
  cv2.imshow("AI Hand Gesture Recognition", image)

  # Press 'q' key to exit the loop
  if cv2.waitKey(5) & 0xFF == ord("q"):
    break

cap.release()
cv2.destroyAllWindows()
