#!/usr/bin/env python3
"""
Main entry point for training ASL classifier.
Trains a RandomForest model on processed hand landmark data.
"""

from src.train import load_processed_data, train_model, save_model
import config
import sys
import os


def main():
    """Train ASL classifier model."""
    # Check if processed data exists
    if not os.path.exists(config.PROCESSED_DATA_FILE):
        print(f"\n✗ Error: Processed data not found at {config.PROCESSED_DATA_FILE}")
        print("\nPlease run data processing first:")
        print("  python process_data.py")
        sys.exit(1)
    
    try:
        # Load processed data
        print("Loading processed data...")
        data, labels = load_processed_data()
        
        # Train model
        model, label_encoder, accuracy = train_model(data, labels)
        
        # Save model
        save_model(model, label_encoder)
        
        print("\n✓ Training complete!")
        print(f"  - Test accuracy: {accuracy * 100:.2f}%")
        print(f"  - Model saved to: {config.MODEL_FILE}")
        print(f"  - Label encoder saved to: {config.LABEL_ENCODER_FILE}")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
