"""
Test script for evaluating trained ASL model.
Generates confusion matrix, classification report, and ROC curves.
"""

import os
import sys
import cv2
import pickle
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import cycle

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import config
from src.data_processor import extract_hand_landmarks


def evaluate_model(model_path=None, encoder_path=None, test_data_dir=None):
    """
    Evaluate trained model on test dataset.
    
    Args:
        model_path: Path to trained model (defaults to config.MODEL_FILE)
        encoder_path: Path to label encoder (defaults to config.LABEL_ENCODER_FILE)
        test_data_dir: Path to test dataset directory
    """
    if model_path is None:
        model_path = config.MODEL_FILE
    if encoder_path is None:
        encoder_path = config.LABEL_ENCODER_FILE
    if test_data_dir is None:
        test_data_dir = config.ASL_TEST_DIR
    
    # Load model
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    model = model_data['model']
    
    # Load label encoder
    with open(encoder_path, 'rb') as f:
        encoder_data = pickle.load(f)
    label_encoder = encoder_data['label_encoder']
    
    # Load test data
    print(f"Loading test data from: {test_data_dir}")
    test_data = []
    test_labels = []
    
    if not os.path.exists(test_data_dir):
        print(f"Warning: Test directory not found: {test_data_dir}")
        print("Please update config.py with correct test data path")
        return
    
    # Process test images
    image_files = [f for f in os.listdir(test_data_dir)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"Found {len(image_files)} test images")
    
    for img_file in image_files:
        img_path = os.path.join(test_data_dir, img_file)
        
        # Extract landmarks
        landmarks = extract_hand_landmarks(img_path)
        
        if landmarks is not None and len(landmarks) == 42:
            test_data.append(landmarks)
            # Extract label from filename (assuming format "A_test.jpg")
            label = img_file.split('_')[0]
            test_labels.append(label)
    
    if len(test_data) == 0:
        print("No test data could be processed. Check dataset path and image format.")
        return
    
    print(f"Processed {len(test_data)} test samples")
    
    # Convert to numpy arrays
    test_data = np.asarray(test_data)
    test_labels = np.asarray(test_labels)
    
    # Encode labels
    try:
        test_labels_encoded = label_encoder.transform(test_labels)
    except ValueError as e:
        print(f"Error encoding labels: {e}")
        print("Some test labels may not be in training set")
        return
    
    # Predict
    print("Making predictions...")
    y_predict = model.predict(test_data)
    
    # Calculate accuracy
    accuracy = accuracy_score(test_labels_encoded, y_predict)
    print(f'\nAccuracy: {accuracy * 100:.2f}%')
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(
        test_labels_encoded,
        y_predict,
        target_names=label_encoder.classes_,
        labels=np.unique(y_predict)
    ))
    
    # Confusion matrix
    cm = confusion_matrix(test_labels_encoded, y_predict)
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    cm_path = os.path.join(config.MODELS_DIR, 'confusion_matrix.png')
    plt.savefig(cm_path)
    print(f"Confusion matrix saved to: {cm_path}")
    plt.close()
    
    # ROC curves (for multi-class)
    n_classes = len(label_encoder.classes_)
    if n_classes > 2:
        y_test_bin = label_binarize(test_labels_encoded, classes=np.arange(n_classes))
        y_predict_bin = label_binarize(y_predict, classes=np.arange(n_classes))
        
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_predict_bin[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        plt.figure(figsize=(10, 8))
        colors = cycle([
            'aqua', 'darkorange', 'cornflowerblue', 'green', 'red',
            'purple', 'cyan', 'magenta', 'yellow', 'black'
        ])
        
        for i, color in zip(range(min(n_classes, 10)), colors):
            plt.plot(
                fpr[i],
                tpr[i],
                color=color,
                lw=2,
                label=f'{label_encoder.classes_[i]} (AUC = {roc_auc[i]:.2f})'
            )
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for ASL Alphabet Classification')
        plt.legend(loc="lower right", fontsize=8)
        plt.grid(True)
        
        roc_path = os.path.join(config.MODELS_DIR, 'roc_curves.png')
        plt.savefig(roc_path)
        print(f"ROC curves saved to: {roc_path}")
        plt.close()
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    evaluate_model()
