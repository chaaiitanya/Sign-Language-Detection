import cv2
import pickle
import numpy as np
import mediapipe as mp

# Load the trained model and label encoder
with open('asl_model_v2.p', 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
label_encoder = model_data['label_encoder_v2']

# Initialize MediaPipe Hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, max_num_hands=2)

# Initialize video capture
cap = cv2.VideoCapture(0)  # Use 0 for the default camera

# Function to preprocess frame for ASL prediction
def preprocess_frame(frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            data_aux = []
            x_ = []
            y_ = []
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            # Normalize and prepare data for prediction
            for i in range(len(hand_landmarks.landmark)):
                data_aux.append(x_[i] - min(x_))
                data_aux.append(y_[i] - min(y_))

            # Ensure the data is the correct length
            if len(data_aux) == 42:  # 21 landmarks with (x, y)
                return np.array(data_aux).reshape(1, -1)

    return None

# Start real-time detection
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Get the processed data for prediction
    data_for_prediction = preprocess_frame(frame)

    if data_for_prediction is not None:
        # Predict the label
        prediction = model.predict(data_for_prediction)
        predicted_label = label_encoder.inverse_transform(prediction)

        # Display the prediction on the frame
        cv2.putText(frame, f'Predicted: {predicted_label[0]}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('ASL Detection', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
