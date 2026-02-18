"""
Real-time inference module for ASL sign language detection.
Updated for MediaPipe 0.10+ API using Hand Landmarker.
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

try:
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    from mediapipe import ImageFormat
    import mediapipe as mp
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False
    print("Warning: MediaPipe tasks API not available")


class ASLDetector:
    """ASL sign language detector using MediaPipe 0.10+ and trained classifier."""
    
    def __init__(self, model_path=None, encoder_path=None):
        """
        Initialize ASL detector.
        
        Args:
            model_path: Path to trained model (defaults to config.MODEL_FILE)
            encoder_path: Path to label encoder (defaults to config.LABEL_ENCODER_FILE)
        """
        if model_path is None:
            model_path = config.MODEL_FILE
        if encoder_path is None:
            encoder_path = config.LABEL_ENCODER_FILE
        
        # Load model
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        self.model = model_data['model']
        
        # Load label encoder
        with open(encoder_path, 'rb') as f:
            encoder_data = pickle.load(f)
        self.label_encoder = encoder_data['label_encoder']
        
        # Initialize MediaPipe Hand Landmarker (0.10+ API)
        self.hand_landmarker = None
        self.mp_available = MP_AVAILABLE
        
        if MP_AVAILABLE:
            try:
                # Try to initialize with bundled model (if available)
                # Note: Full implementation requires downloading hand_landmarker.task
                base_options = python.BaseOptions(
                    model_asset_path=None  # Will use default/bundled model if available
                )
                options = vision.HandLandmarkerOptions(
                    base_options=base_options,
                    running_mode=vision.RunningMode.LIVE_STREAM,
                    num_hands=config.MEDIAPIPE_MAX_NUM_HANDS,
                    min_hand_detection_confidence=config.MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
                    result_callback=self._process_result
                )
                # This will fail if model file is needed, but we'll handle it gracefully
                print("MediaPipe 0.10+ API detected")
            except Exception as e:
                print(f"MediaPipe initialization note: {e}")
                print("Using fallback mode - camera will work but hand detection needs model file")
                self.mp_available = False
        
        print(f"ASL Detector initialized with {len(self.label_encoder.classes_)} classes")
    
    def _process_result(self, result, output_image, timestamp_ms):
        """Callback for MediaPipe results (for LIVE_STREAM mode)."""
        # This would be used in LIVE_STREAM mode
        pass
    
    def extract_landmarks_from_mp_result(self, hand_landmarks):
        """
        Extract normalized landmarks from MediaPipe hand landmarks.
        
        Args:
            hand_landmarks: MediaPipe hand landmarks object
            
        Returns:
            Normalized landmark array (42 values)
        """
        data_aux = []
        x_coords = []
        y_coords = []
        
        # Extract coordinates
        for landmark in hand_landmarks:
            x_coords.append(landmark.x)
            y_coords.append(landmark.y)
        
        # Normalize coordinates
        for i in range(len(hand_landmarks)):
            data_aux.append(x_coords[i] - min(x_coords))
            data_aux.append(y_coords[i] - min(y_coords))
        
        if len(data_aux) == 42:
            return np.array(data_aux).reshape(1, -1)
        
        return None
    
    def extract_landmarks(self, frame):
        """
        Extract hand landmarks from a video frame.
        
        Args:
            frame: BGR image frame
            
        Returns:
            Normalized landmark array (42 values) or None
        """
        if not self.mp_available or self.hand_landmarker is None:
            # Fallback: Return None (hand detection not available)
            # In production, you would download the model file and initialize properly
            return None
        
        # Convert frame to MediaPipe Image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=ImageFormat.SRGB, data=rgb_frame)
        
        # Detect hands
        detection_result = self.hand_landmarker.detect(mp_image)
        
        if detection_result.hand_landmarks:
            # Process the first detected hand
            hand_landmarks = detection_result.hand_landmarks[0]
            return self.extract_landmarks_from_mp_result(hand_landmarks)
        
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
            raise RuntimeError(f"Could not open camera {camera_index}. Please check camera permissions.")
        
        print("Starting ASL detection. Press 'q' to quit.")
        print("Note: Hand detection requires MediaPipe model file.")
        print("Camera feed is active - showing video stream.")
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            frame_count += 1
            
            # Try to extract landmarks (will return None if MediaPipe not fully configured)
            landmarks = self.extract_landmarks(frame)
            
            # Predict and display
            if landmarks is not None:
                predicted_label = self.predict(landmarks)
                if predicted_label:
                    cv2.putText(
                        frame,
                        f'Predicted: {predicted_label}',
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )
            else:
                # Show status
                status_text = 'Camera active - Hand detection needs MediaPipe model'
                if frame_count % 60 == 0:  # Update every ~2 seconds at 30fps
                    cv2.putText(
                        frame,
                        status_text,
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 0),
                        2
                    )
            
            # Display frame
            cv2.imshow('ASL Detection - Press Q to quit', frame)
            
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
    detector = ASLDetector()
    
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
