"""
Real-time inference module for ASL sign language detection.
Uses webcam to detect and classify ASL signs in real-time.
Updated for MediaPipe 0.10+ API.
"""

import os
import sys
import cv2
import pickle
import numpy as np

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import config

# Import MediaPipe 0.10+ API
try:
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    from mediapipe import ImageFormat
    import mediapipe as mp
    NEW_API_AVAILABLE = True
except ImportError:
    NEW_API_AVAILABLE = False
    raise ImportError("MediaPipe 0.10+ is required. Install with: pip install mediapipe")


class ASLDetector:
    """ASL sign language detector using MediaPipe 0.10+ and trained classifier."""
    
    def __init__(self, model_path=None, encoder_path=None, hand_model_path=None):
        """
        Initialize ASL detector.
        
        Args:
            model_path: Path to trained model (defaults to config.MODEL_FILE)
            encoder_path: Path to label encoder (defaults to config.LABEL_ENCODER_FILE)
            hand_model_path: Path to MediaPipe hand landmarker model (defaults to models/hand_landmarker.task)
        """
        if model_path is None:
            model_path = config.MODEL_FILE
        if encoder_path is None:
            encoder_path = config.LABEL_ENCODER_FILE
        if hand_model_path is None:
            hand_model_path = os.path.join(config.MODELS_DIR, 'hand_landmarker.task')
        
        # Load ASL classification model
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        self.model = model_data['model']
        
        # Load label encoder
        with open(encoder_path, 'rb') as f:
            encoder_data = pickle.load(f)
        self.label_encoder = encoder_data['label_encoder']
        
        # Initialize MediaPipe Hand Landmarker (0.10+ API)
        if not os.path.exists(hand_model_path):
            raise FileNotFoundError(
                f"MediaPipe hand landmarker model not found at {hand_model_path}\n"
                f"Download it with: curl -L -o {hand_model_path} "
                f"https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            )
        
        base_options = python.BaseOptions(model_asset_path=hand_model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,  # Use VIDEO mode for webcam
            num_hands=config.MEDIAPIPE_MAX_NUM_HANDS,
            min_hand_detection_confidence=config.MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.hand_landmarker = vision.HandLandmarker.create_from_options(options)
        
        # Drawing utilities
        self.mp_drawing = mp.tasks.vision.drawing_utils
        self.mp_drawing_styles = mp.tasks.vision.drawing_styles
        self.mp_hands = mp.tasks.vision.HandLandmarksConnections
        
        print(f"✓ ASL Detector initialized with {len(self.label_encoder.classes_)} classes")
        print(f"✓ MediaPipe Hand Landmarker loaded from {hand_model_path}")
    
    def extract_landmarks_from_result(self, hand_landmarks):
        """
        Extract normalized landmarks from MediaPipe hand landmarks.
        
        Args:
            hand_landmarks: List of MediaPipe hand landmarks
            
        Returns:
            Normalized landmark array (42 values) or None
        """
        if not hand_landmarks or len(hand_landmarks) == 0:
            return None
        
        # Use the first detected hand
        landmarks = hand_landmarks[0]
        data_aux = []
        x_coords = []
        y_coords = []
        
        # Extract coordinates
        for landmark in landmarks:
            x_coords.append(landmark.x)
            y_coords.append(landmark.y)
        
        # Normalize coordinates relative to minimum
        for i in range(len(landmarks)):
            data_aux.append(x_coords[i] - min(x_coords))
            data_aux.append(y_coords[i] - min(y_coords))
        
        if len(data_aux) == 42:
            return np.array(data_aux).reshape(1, -1)
        
        return None
    
    def extract_landmarks(self, frame, timestamp_ms=0):
        """
        Extract hand landmarks from a video frame.
        
        Args:
            frame: BGR image frame
            timestamp_ms: Timestamp in milliseconds (for VIDEO mode)
            
        Returns:
            Normalized landmark array (42 values) or None
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image
        mp_image = mp.Image(image_format=ImageFormat.SRGB, data=rgb_frame)
        
        # Detect hands
        detection_result = self.hand_landmarker.detect_for_video(mp_image, timestamp_ms)
        
        # Extract landmarks
        if detection_result.hand_landmarks:
            return self.extract_landmarks_from_result(detection_result.hand_landmarks)
        
        return None
    
    def predict(self, landmarks):
        """
        Predict ASL sign from hand landmarks.
        
        Args:
            landmarks: Normalized landmark array
            
        Returns:
            Predicted label string or None
        """
        if landmarks is None:
            return None
        
        prediction = self.model.predict(landmarks)
        predicted_label = self.label_encoder.inverse_transform(prediction)[0]
        return predicted_label
    
    def run(self, camera_index=0, show_landmarks=False):
        """
        Run real-time ASL detection from webcam.
        
        Args:
            camera_index: Camera device index (default: 0)
            show_landmarks: Whether to draw hand landmarks on frame
        """
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open camera {camera_index}. "
                "Please check camera permissions in System Preferences."
            )
        
        # Get camera FPS for timestamp calculation
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30  # Default to 30 FPS
        
        frame_time_ms = int(1000 / fps)
        timestamp_ms = 0
        
        print("Starting ASL detection. Press 'q' to quit.")
        if show_landmarks:
            print("Hand landmarks visualization enabled.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Convert frame to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=ImageFormat.SRGB, data=rgb_frame)
            
            # Detect hands
            detection_result = self.hand_landmarker.detect_for_video(mp_image, timestamp_ms)
            
            # Draw landmarks if requested
            if show_landmarks and detection_result.hand_landmarks:
                annotated_image = np.copy(rgb_frame)
                for hand_landmarks in detection_result.hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        annotated_image,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )
                frame = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
            
            # Extract landmarks and predict
            landmarks = self.extract_landmarks_from_result(detection_result.hand_landmarks)
            
            if landmarks is not None:
                predicted_label = self.predict(landmarks)
                if predicted_label:
                    # Display prediction
                    cv2.putText(
                        frame,
                        f'Predicted: {predicted_label}',
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )
                    
                    # Show confidence or hand count
                    hand_count = len(detection_result.hand_landmarks)
                    cv2.putText(
                        frame,
                        f'Hands detected: {hand_count}',
                        (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 0),
                        2
                    )
            else:
                cv2.putText(
                    frame,
                    'Show your hand to the camera',
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 165, 255),
                    2
                )
            
            # Display frame
            cv2.imshow('ASL Detection - Press Q to quit', frame)
            
            # Update timestamp
            timestamp_ms += frame_time_ms
            
            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("ASL detection stopped.")


if __name__ == "__main__":
    import sys
    
    # Check if model files exist
    if not os.path.exists(config.MODEL_FILE):
        print(f"Error: Model file not found at {config.MODEL_FILE}")
        print("Please train the model first using: python train_model.py")
        sys.exit(1)
    
    # Initialize and run detector
    try:
        detector = ASLDetector()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error initializing detector: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Parse command line arguments
    show_landmarks = '--landmarks' in sys.argv or '-l' in sys.argv
    
    try:
        detector.run(show_landmarks=show_landmarks)
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
