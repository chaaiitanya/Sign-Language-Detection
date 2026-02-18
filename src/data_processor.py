"""
Data processing module for ASL sign language detection.
Extracts hand landmarks using MediaPipe from ASL alphabet images.
"""

import os
import sys
import pickle
import cv2
import mediapipe as mp
from tqdm import tqdm

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import config


def extract_hand_landmarks(image_path):
    """
    Extract hand landmarks from an image using MediaPipe.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        List of normalized landmark coordinates (42 values: 21 landmarks * 2 coords)
        or None if no hand detected
    """
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # Convert to RGB (required by MediaPipe)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Initialize MediaPipe Hands (static mode for images)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        min_detection_confidence=config.MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
        max_num_hands=config.MEDIAPIPE_MAX_NUM_HANDS
    )
    
    # Process the image
    results = hands.process(img_rgb)
    
    if results.multi_hand_landmarks:
        # Process the first detected hand
        hand_landmarks = results.multi_hand_landmarks[0]
        data_aux = []
        x_coords = []
        y_coords = []
        
        # Collect all coordinates
        for landmark in hand_landmarks.landmark:
            x_coords.append(landmark.x)
            y_coords.append(landmark.y)
        
        # Normalize coordinates relative to minimum values
        for i in range(len(hand_landmarks.landmark)):
            data_aux.append(x_coords[i] - min(x_coords))
            data_aux.append(y_coords[i] - min(y_coords))
        
        hands.close()
        return data_aux
    
    hands.close()
    return None


def process_dataset(data_dir=None):
    """
    Process ASL alphabet dataset and extract hand landmarks.
    
    Args:
        data_dir: Path to dataset directory (defaults to config.ASL_TRAIN_DIR)
        
    Returns:
        Tuple of (data, labels) lists
    """
    if data_dir is None:
        data_dir = config.ASL_TRAIN_DIR
    
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
    
    data = []
    labels = []
    
    print(f"Processing dataset from: {data_dir}")
    
    # Get all label directories
    label_dirs = [d for d in os.listdir(data_dir) 
                  if os.path.isdir(os.path.join(data_dir, d))]
    label_dirs.sort()
    
    # Process each label directory
    for label in tqdm(label_dirs, desc="Processing labels"):
        label_path = os.path.join(data_dir, label)
        
        # Process each image in the label directory
        image_files = [f for f in os.listdir(label_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for img_file in image_files:
            img_path = os.path.join(label_path, img_file)
            
            # Extract hand landmarks
            landmarks = extract_hand_landmarks(img_path)
            
            if landmarks is not None and len(landmarks) == 42:
                data.append(landmarks)
                labels.append(label)
    
    print(f"Processed {len(data)} samples with {len(set(labels))} unique labels")
    return data, labels


def save_processed_data(data, labels, output_path=None):
    """
    Save processed data to pickle file.
    
    Args:
        data: List of landmark data
        labels: List of labels
        output_path: Output file path (defaults to config.PROCESSED_DATA_FILE)
    """
    if output_path is None:
        output_path = config.PROCESSED_DATA_FILE
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump({'data': data, 'labels': labels}, f)
    
    print(f"Processed data saved to: {output_path}")


def load_processed_data(input_path=None):
    """
    Load processed data from pickle file.
    
    Args:
        input_path: Input file path (defaults to config.PROCESSED_DATA_FILE)
        
    Returns:
        Tuple of (data, labels) lists
    """
    if input_path is None:
        input_path = config.PROCESSED_DATA_FILE
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Processed data file not found: {input_path}")
    
    with open(input_path, 'rb') as f:
        data_dict = pickle.load(f)
    
    return data_dict['data'], data_dict['labels']


if __name__ == "__main__":
    # Process dataset and save
    data, labels = process_dataset()
    save_processed_data(data, labels)
    print("Data processing complete!")
