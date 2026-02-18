import cv2
import numpy as np
import pickle

# Load the trained model
with open('model.p', 'rb') as f:
    model_data = pickle.load(f)
model = model_data['model']

# Function to predict the letter based on landmarks
def predict_letter(landmark_list):
    # Ensure the landmark_list has the correct size
    if len(landmark_list) != 16:  # Assuming you're using 16 landmarks for 48 features
        print(f"Incorrect number of landmarks: {len(landmark_list)}")
        return None  # or handle this case appropriately

    input_data = np.array(landmark_list).flatten().reshape(1, -1)  # Flatten and reshape
    print(f"Input data shape: {input_data.shape}")  # Debug line

    # Predict with the correct input shape
    prediction = model.predict(input_data)
    print(f'Predicted letter (index): {prediction[0]}')
    return chr(prediction[0] + ord('A'))  # Adjust based on your label encoding

# Capture video from webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Add code here to process the frame and extract hand landmarks
    # landmark_list should be populated with the current hand landmarks
    landmark_list = []  # Replace this with actual landmark extraction logic

    # Predict the letter based on landmarks
    predicted_letter = predict_letter(landmark_list)
    if predicted_letter:
        print(f"Predicted Letter: {predicted_letter}")

    cv2.imshow("Webcam", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
