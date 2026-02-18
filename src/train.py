"""
Training module for ASL sign language classifier.
Trains a RandomForest classifier on MediaPipe hand landmark data.
"""

import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import sys

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import config
from src.data_processor import load_processed_data


def train_model(data, labels, n_estimators=None, max_depth=None, test_size=None):
    """
    Train a RandomForest classifier on ASL hand landmark data.
    
    Args:
        data: List of hand landmark features
        labels: List of corresponding labels
        n_estimators: Number of trees (defaults to config.N_ESTIMATORS)
        max_depth: Maximum depth of trees (defaults to config.MAX_DEPTH)
        test_size: Test split ratio (defaults to config.TRAIN_TEST_SPLIT)
        
    Returns:
        Tuple of (trained_model, label_encoder, test_accuracy)
    """
    if n_estimators is None:
        n_estimators = config.N_ESTIMATORS
    if max_depth is None:
        max_depth = config.MAX_DEPTH
    if test_size is None:
        test_size = config.TRAIN_TEST_SPLIT
    
    # Filter data to ensure consistent feature length (42 for single hand)
    # Some entries may have 84 features (two hands), we'll use only single-hand data
    filtered_data = []
    filtered_labels = []
    for d, l in zip(data, labels):
        if len(d) == 42:  # Single hand (21 landmarks * 2 coordinates)
            filtered_data.append(d)
            filtered_labels.append(l)
        elif len(d) == 84:  # Two hands - take first hand only
            filtered_data.append(d[:42])
            filtered_labels.append(l)
    
    if len(filtered_data) == 0:
        raise ValueError("No valid data found. Expected 42 or 84 features per sample.")
    
    print(f"Filtered {len(filtered_data)} samples with consistent feature length (42)")
    if len(data) != len(filtered_data):
        print(f"Removed {len(data) - len(filtered_data)} samples with inconsistent lengths")
    
    # Convert to numpy arrays
    data = np.asarray(filtered_data)
    labels = np.asarray(filtered_labels)
    
    # Encode labels
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    
    print(f"Training on {len(data)} samples")
    print(f"Number of classes: {len(label_encoder.classes_)}")
    print(f"Classes: {', '.join(label_encoder.classes_)}")
    
    # Split data
    x_train, x_test, y_train, y_test = train_test_split(
        data, labels_encoded, 
        test_size=test_size, 
        shuffle=True, 
        stratify=labels_encoded,
        random_state=config.RANDOM_STATE
    )
    
    print(f"Training set: {len(x_train)} samples")
    print(f"Test set: {len(x_test)} samples")
    
    # Initialize model for learning curve (using n_jobs=1 to avoid sandbox issues)
    curve_model = RandomForestClassifier(
        n_estimators=10,  # Small number for learning curve
        max_depth=max_depth,
        random_state=config.RANDOM_STATE,
        n_jobs=1  # Use single job to avoid sandbox restrictions
    )
    
    # Generate learning curves (using n_jobs=1 to avoid sandbox issues)
    print("Generating learning curves...")
    try:
        train_sizes, train_scores, val_scores = learning_curve(
            curve_model, x_train, y_train,
            cv=3,  # Reduced from 5 to avoid issues with small classes
            n_jobs=1,  # Use single job to avoid sandbox restrictions
            train_sizes=np.linspace(0.1, 1.0, 5),  # Reduced points for faster execution
            scoring='accuracy'
        )
        
        # Calculate statistics
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        # Plot learning curves
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, 'o-', color="r", label="Training score")
        plt.plot(train_sizes, val_mean, 'o-', color="g", label="Cross-validation score")
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                         color="r", alpha=0.1)
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, 
                         color="g", alpha=0.1)
        plt.title('Learning Curve')
        plt.xlabel('Training Set Size')
        plt.ylabel('Accuracy')
        plt.legend(loc='best')
        plt.grid(True)
        
        # Save learning curve
        learning_curve_path = os.path.join(config.MODELS_DIR, 'learning_curve.png')
        plt.savefig(learning_curve_path)
        print(f"Learning curve saved to: {learning_curve_path}")
        plt.close()
    except Exception as e:
        print(f"Warning: Could not generate learning curve: {e}")
        print("Continuing with model training...")
    
    # Initialize model for actual training (using n_jobs=1 to avoid sandbox issues)
    model = RandomForestClassifier(
        n_estimators=0,
        max_depth=max_depth,
        warm_start=True,
        random_state=config.RANDOM_STATE,
        n_jobs=1  # Use single job to avoid sandbox restrictions
    )
    
    # Train model with progress bar
    print(f"Training model with {n_estimators} trees...")
    for i in tqdm(range(n_estimators), desc="Training Progress"):
        model.n_estimators += 1
        model.fit(x_train, y_train)
    
    # Evaluate on test set
    y_predict = model.predict(x_test)
    test_accuracy = accuracy_score(y_test, y_predict)
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")
    
    return model, label_encoder, test_accuracy


def save_model(model, label_encoder, model_path=None, encoder_path=None):
    """
    Save trained model and label encoder.
    
    Args:
        model: Trained RandomForest model
        label_encoder: Fitted LabelEncoder
        model_path: Path to save model (defaults to config.MODEL_FILE)
        encoder_path: Path to save encoder (defaults to config.LABEL_ENCODER_FILE)
    """
    if model_path is None:
        model_path = config.MODEL_FILE
    if encoder_path is None:
        encoder_path = config.LABEL_ENCODER_FILE
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Save model
    with open(model_path, 'wb') as f:
        pickle.dump({'model': model}, f)
    print(f"Model saved to: {model_path}")
    
    # Save label encoder
    with open(encoder_path, 'wb') as f:
        pickle.dump({'label_encoder': label_encoder}, f)
    print(f"Label encoder saved to: {encoder_path}")


if __name__ == "__main__":
    import os
    
    # Load processed data
    print("Loading processed data...")
    data, labels = load_processed_data()
    
    # Train model
    model, label_encoder, accuracy = train_model(data, labels)
    
    # Save model
    save_model(model, label_encoder)
    
    print("Training complete!")
