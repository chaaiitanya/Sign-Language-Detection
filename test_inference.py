#!/usr/bin/env python3
"""
Test inference script that verifies the model loads and can make predictions.
Works without camera access.
"""

import numpy as np
import pickle
import os
import config
from src.inference import ASLDetector


def test_model_loading():
    """Test that model and encoder load correctly."""
    print("Testing model loading...")
    
    # Check files exist
    if not os.path.exists(config.MODEL_FILE):
        print(f"✗ Model file not found: {config.MODEL_FILE}")
        return False
    
    if not os.path.exists(config.LABEL_ENCODER_FILE):
        print(f"✗ Label encoder file not found: {config.LABEL_ENCODER_FILE}")
        return False
    
    # Load model
    with open(config.MODEL_FILE, 'rb') as f:
        model_data = pickle.load(f)
    model = model_data['model']
    
    # Load encoder
    with open(config.LABEL_ENCODER_FILE, 'rb') as f:
        encoder_data = pickle.load(f)
    label_encoder = encoder_data['label_encoder']
    
    print(f"✓ Model loaded successfully")
    print(f"✓ Label encoder loaded with {len(label_encoder.classes_)} classes")
    print(f"  Classes: {', '.join(label_encoder.classes_[:10])}...")
    
    return True, model, label_encoder


def test_prediction(model, label_encoder):
    """Test making predictions with sample data."""
    print("\nTesting predictions...")
    
    # Create sample hand landmark data (42 features: 21 landmarks * 2 coords)
    # This simulates normalized hand landmarks
    np.random.seed(42)
    sample_landmarks = np.random.rand(1, 42) * 0.1  # Small normalized values
    
    try:
        prediction = model.predict(sample_landmarks)
        predicted_label = label_encoder.inverse_transform(prediction)[0]
        
        print(f"✓ Prediction successful")
        print(f"  Sample input shape: {sample_landmarks.shape}")
        print(f"  Predicted label: {predicted_label}")
        
        # Test with multiple samples
        multiple_samples = np.random.rand(5, 42) * 0.1
        predictions = model.predict(multiple_samples)
        predicted_labels = label_encoder.inverse_transform(predictions)
        
        print(f"✓ Batch prediction successful")
        print(f"  Predicted labels: {', '.join(predicted_labels)}")
        
        return True
    except Exception as e:
        print(f"✗ Prediction failed: {e}")
        return False


def test_detector_initialization():
    """Test ASLDetector initialization."""
    print("\nTesting ASLDetector initialization...")
    
    try:
        detector = ASLDetector()
        print(f"✓ ASLDetector initialized successfully")
        print(f"  Model classes: {len(detector.label_encoder.classes_)}")
        return True, detector
    except Exception as e:
        print(f"✗ ASLDetector initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def main():
    """Run all tests."""
    print("=" * 60)
    print("ASL Detection Project - Inference Test")
    print("=" * 60)
    
    # Test 1: Model loading
    result = test_model_loading()
    if not result:
        print("\n✗ Model loading test failed. Cannot continue.")
        return
    
    success, model, label_encoder = result
    
    # Test 2: Predictions
    if not test_prediction(model, label_encoder):
        print("\n✗ Prediction test failed.")
        return
    
    # Test 3: Detector initialization
    success, detector = test_detector_initialization()
    if not success:
        print("\n✗ Detector initialization test failed.")
        return
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
    print("\nProject Status:")
    print("  ✓ Model trained and saved")
    print("  ✓ Model loads correctly")
    print("  ✓ Predictions work")
    print("  ✓ ASLDetector initializes")
    print("\nTo run real-time detection:")
    print("  1. Grant camera permissions in System Preferences")
    print("  2. Run: python3 run_inference.py")
    print("  3. Or with landmarks: python3 run_inference.py --landmarks")


if __name__ == "__main__":
    main()
