#!/usr/bin/env python3
"""
Main entry point for processing ASL dataset.
Extracts hand landmarks from images using MediaPipe.
"""

from src.data_processor import process_dataset, save_processed_data
import config
import sys


def main():
    """Process ASL dataset and save processed data."""
    try:
        # Process dataset
        data, labels = process_dataset()
        
        # Save processed data
        save_processed_data(data, labels)
        
        print("\n✓ Data processing complete!")
        print(f"  - Processed {len(data)} samples")
        print(f"  - Found {len(set(labels))} unique labels")
        print(f"  - Saved to: {config.PROCESSED_DATA_FILE}")
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("\nPlease update config.py with the correct dataset path.")
        print("Dataset should be organized as:")
        print("  data/raw/asl_alphabet_train/asl_alphabet_train/<letter>/<images>")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
