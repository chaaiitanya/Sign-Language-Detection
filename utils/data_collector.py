"""
Utility script for collecting custom ASL sign data from webcam.
Creates dataset directories and captures images for each class.
"""

import os
import cv2
import config


def collect_data(number_of_classes=26, dataset_size=100, data_dir=None):
    """
    Collect ASL sign data from webcam.
    
    Args:
        number_of_classes: Number of ASL classes to collect
        dataset_size: Number of images per class
        data_dir: Directory to save data (defaults to config.RAW_DATA_DIR)
    """
    if data_dir is None:
        data_dir = config.RAW_DATA_DIR
    
    # Create data directory
    os.makedirs(data_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")
    
    for j in range(number_of_classes):
        class_dir = os.path.join(data_dir, str(j))
        os.makedirs(class_dir, exist_ok=True)
        
        print(f'Collecting data for class {j}')
        
        # Wait for user to be ready
        done = False
        while True:
            ret, frame = cap.read()
            cv2.putText(
                frame,
                f'Ready for class {j}? Press "Q"!',
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.3,
                (0, 255, 0),
                3,
                cv2.LINE_AA
            )
            cv2.imshow('frame', frame)
            if cv2.waitKey(25) == ord('q'):
                break
        
        # Collect images
        counter = 0
        while counter < dataset_size:
            ret, frame = cap.read()
            cv2.putText(
                frame,
                f'Collecting: {counter}/{dataset_size}',
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            cv2.imshow('frame', frame)
            cv2.waitKey(25)
            cv2.imwrite(
                os.path.join(class_dir, f'{counter}.jpg'),
                frame
            )
            counter += 1
    
    cap.release()
    cv2.destroyAllWindows()
    print("Data collection complete!")


if __name__ == "__main__":
    import sys
    
    # Parse arguments
    n_classes = 26
    n_samples = 100
    
    if len(sys.argv) > 1:
        n_classes = int(sys.argv[1])
    if len(sys.argv) > 2:
        n_samples = int(sys.argv[2])
    
    print(f"Collecting {n_samples} samples for {n_classes} classes")
    collect_data(n_classes, n_samples)
