#!/usr/bin/env python3
"""
Main entry point for real-time ASL detection.
Runs inference on webcam feed.
"""

from src.inference import ASLDetector
import config
import sys
import os


def main():
    """Run real-time ASL detection."""
    # Check if model exists
    if not os.path.exists(config.MODEL_FILE):
        print(f"\n✗ Error: Model not found at {config.MODEL_FILE}")
        print("\nPlease train the model first:")
        print("  python train_model.py")
        sys.exit(1)
    
    if not os.path.exists(config.LABEL_ENCODER_FILE):
        print(f"\n✗ Error: Label encoder not found at {config.LABEL_ENCODER_FILE}")
        print("\nPlease train the model first:")
        print("  python train_model.py")
        sys.exit(1)
    
    try:
        # Initialize detector
        detector = ASLDetector()
        
        # Parse arguments
        show_landmarks = '--landmarks' in sys.argv or '-l' in sys.argv
        
        # Run detection
        detector.run(show_landmarks=show_landmarks)
        
    except KeyboardInterrupt:
        print("\n\nStopped by user")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
