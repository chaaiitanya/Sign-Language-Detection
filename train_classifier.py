import os
import cv2
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # Library to show progress bars

print("Starting script...")

# Define the path to the dataset directory
data_dir = './asl_alphabet_train/asl_alphabet_train'

# Initialize lists for data and labels
data = []
labels = []

print("Checking directories and reading images...")

# Loop through each folder (which represents a letter or class)
for label in os.listdir(data_dir):
    folder_path = os.path.join(data_dir, label)

    print(f"Processing folder: {label}")

    # Check if it is a directory (to avoid any non-folder files)
    if os.path.isdir(folder_path):
        for img_file in os.listdir(folder_path):
            # Full path to the image file
            img_path = os.path.join(folder_path, img_file)

            print(f"Reading image: {img_file} from {label}")

            # Read the image using OpenCV (or any other image reading library)
            img = cv2.imread(img_path)

            # Check if image was read successfully
            if img is None:
                print(f"Warning: Image {img_file} could not be loaded from {label}")
                continue

            # Resize the image for uniformity, for example 64x64
            img_resized = cv2.resize(img, (64, 64))

            # Flatten the image into a single-dimensional array (for simple models)
            img_flattened = img_resized.flatten()

            # Append the flattened image and label to the lists
            data.append(img_flattened)
            labels.append(label)

print("Finished reading images. Total samples loaded:", len(data))

# Convert lists to numpy arrays
data = np.asarray(data)
labels = np.asarray(labels)

# Encode the labels (letters) to numeric values
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Split the data into training and testing sets (80% training, 20% testing)
x_train, x_test, y_train, y_test = train_test_split(data, labels_encoded, test_size=0.2, shuffle=True, stratify=labels_encoded)

# Initialize the RandomForestClassifier with warm_start=True to add trees incrementally
model = RandomForestClassifier(n_estimators=0, max_depth=20, warm_start=True, random_state=42)

# --- Generate Learning Curves ---
print("Generating learning curves...")

# Using learning_curve to plot training vs cross-validation scores as a function of training size
train_sizes, train_scores, test_scores = learning_curve(
    model, x_train, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10), scoring='accuracy'
)

# Calculate mean and standard deviation for training and test scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Plot learning curves and save as image
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, 'o-', color="r", label="Training score")
plt.plot(train_sizes, test_mean, 'o-', color="g", label="Cross-validation score")
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="r", alpha=0.1)
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="g", alpha=0.1)
plt.title('Learning Curve')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.grid(True)
# Save the plot as an image
plt.savefig('learning_curve.png')
print("Learning curve plot saved as 'learning_curve.png'.")
plt.show()

# Train the model with progress bar
n_estimators = 100  # Number of trees to train
print(f"Training the model with {n_estimators} trees...")

# Use tqdm to display a progress bar during training
for i in tqdm(range(n_estimators), desc="Training Progress"):
    model.n_estimators += 1  # Increment the number of trees by 1
    model.fit(x_train, y_train)  # Fit the model

# Predict the labels on the test set
y_predict = model.predict(x_test)

# Calculate the accuracy of the model
score = accuracy_score(y_test, y_predict)
print(f"Accuracy on test set: {score * 100:.2f}%")

# Save the trained model and label encoder to files with updated names
model_filename = 'asl_model_v2.p'
label_encoder_filename = 'label_encoder_v2.p'

with open(model_filename, 'wb') as f:
    pickle.dump({'model': model}, f)

with open(label_encoder_filename, 'wb') as f:
    pickle.dump({'label_encoder': label_encoder}, f)

print(f"Model saved as {model_filename}")
print(f"Label Encoder saved as {label_encoder_filename}")

print("Script finished.")
