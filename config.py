"""
Configuration file for Sign Language Detection project.
Update paths according to your dataset location.
"""

import os

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Dataset paths (update these to match your dataset location)
ASL_TRAIN_DIR = os.path.join(RAW_DATA_DIR, 'asl_alphabet_train', 'asl_alphabet_train')
ASL_TEST_DIR = os.path.join(RAW_DATA_DIR, 'asl_alphabet_test', 'asl_alphabet_test')

# Model file paths
MODEL_FILE = os.path.join(MODELS_DIR, 'asl_model.pkl')
LABEL_ENCODER_FILE = os.path.join(MODELS_DIR, 'label_encoder.pkl')
PROCESSED_DATA_FILE = os.path.join(PROCESSED_DATA_DIR, 'asl_data.pickle')

# MediaPipe settings
MEDIAPIPE_MIN_DETECTION_CONFIDENCE = 0.5
MEDIAPIPE_MAX_NUM_HANDS = 2
MEDIAPIPE_STATIC_IMAGE_MODE = False

# Training settings
TRAIN_TEST_SPLIT = 0.2
RANDOM_STATE = 42
N_ESTIMATORS = 100
MAX_DEPTH = 20

# Image preprocessing settings
IMAGE_SIZE = (64, 64)

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
