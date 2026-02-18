import os
import cv2
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import cycle

# Path to the test dataset directory (adjust this as necessary)
test_data_dir = './asl_alphabet_test/asl_alphabet_test'

# Load the trained model
with open('asl_model_v2.p', 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']

# Load the label encoder from the separate file
with open('label_encoder_v2.p', 'rb') as f:
    label_encoder_data = pickle.load(f)
    label_encoder = label_encoder_data['label_encoder']

# Initialize lists for test data and labels
test_data = []
test_labels = []

# Load test images from the test dataset directory (individual image files)
print("Checking directories and reading test images...")

for img_file in os.listdir(test_data_dir):
    img_path = os.path.join(test_data_dir, img_file)

    # Only process files that are images (e.g., end in .jpg)
    if img_file.endswith('.jpg'):
        print(f"Reading image: {img_path}")  # Debugging statement

        # Read the image using OpenCV
        img = cv2.imread(img_path)

        # Check if image was read successfully
        if img is None:
            print(f"Warning: Image {img_file} could not be loaded")
            continue

        # Resize the image for uniformity (64x64)
        img_resized = cv2.resize(img, (64, 64))

        # Flatten the image into a single-dimensional array (for simple models)
        img_flattened = img_resized.flatten()

        # Append the flattened image to the data list
        test_data.append(img_flattened)

        # Extract the label from the file name (assuming format "A_test.jpg", "B_test.jpg", etc.)
        label = img_file.split('_')[0]  # Get the letter before the underscore
        test_labels.append(label)

print(f"Finished reading test images. Total test samples loaded: {len(test_data)}")

# Check if data was loaded
if len(test_data) == 0:
    print("No test images were loaded. Please check the test dataset directory.")
    exit()

# Convert test data and labels to numpy arrays
test_data = np.asarray(test_data)
test_labels = np.asarray(test_labels)

# Encode the test labels to numeric values using the label encoder
test_labels_encoded = label_encoder.transform(test_labels)

# --- Predicting on the test set ---
y_predict = model.predict(test_data)

# Calculate the accuracy of the model on the test set
score = accuracy_score(test_labels_encoded, y_predict)
print(f'\n{score * 100:.2f}% of samples were classified correctly on the test set!')

# --- Classification Report ---
print("\nClassification Report:")
print(classification_report(test_labels_encoded, y_predict, target_names=label_encoder.classes_, labels=np.unique(y_predict)))

# --- Confusion Matrix ---
cm = confusion_matrix(test_labels_encoded, y_predict)

# Plotting the confusion matrix
plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
print("Confusion matrix saved as 'confusion_matrix.png'.")
plt.show()

# --- ROC Curve and AUC ---
n_classes = len(label_encoder.classes_)
y_test_bin = label_binarize(test_labels_encoded, classes=np.arange(n_classes))
y_predict_bin = label_binarize(y_predict, classes=np.arange(n_classes))

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_predict_bin[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot all ROC curves
plt.figure(figsize=(10, 8))
colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple', 'cyan', 'magenta', 'yellow', 'black'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'ROC curve of class {label_encoder.classes_[i]} (area = {roc_auc[i]:0.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for ASL Alphabet Classification')
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig('roc_curves.png')
print("ROC curves saved as 'roc_curves.png'.")
plt.show()
