# Sign Language Detection

A machine learning project for real-time American Sign Language (ASL) alphabet recognition using MediaPipe hand tracking and RandomForest classification.

## Features

- **Hand Landmark Extraction**: Uses MediaPipe to extract 21 hand landmarks (42 features) from images
- **Real-time Detection**: Webcam-based real-time ASL sign recognition
- **Model Training**: RandomForest classifier with learning curve visualization
- **Evaluation Tools**: Comprehensive testing with confusion matrix and ROC curves

## Project Structure

```
Sign-Language-Detection/
├── config.py                 # Configuration file (paths, settings)
├── process_data.py           # Main script to process dataset
├── train_model.py            # Main script to train model
├── run_inference.py          # Main script for real-time detection
│
├── src/                      # Source code modules
│   ├── __init__.py
│   ├── data_processor.py    # Data processing utilities
│   ├── train.py             # Training module
│   └── inference.py         # Inference module
│
├── data/                     # Data directories
│   ├── raw/                  # Raw dataset (place your ASL dataset here)
│   └── processed/           # Processed data files
│
├── models/                   # Trained models and outputs
│   ├── asl_model.pkl        # Trained model
│   ├── label_encoder.pkl    # Label encoder
│   ├── learning_curve.png   # Learning curve plot
│   ├── confusion_matrix.png # Confusion matrix
│   └── roc_curves.png       # ROC curves
│
├── tests/                    # Test scripts
│   └── test_model.py        # Model evaluation script
│
├── utils/                    # Utility scripts
│   └── data_collector.py    # Custom data collection tool
│
└── requirements.txt          # Python dependencies
```

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Sign-Language-Detection
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare your dataset**
   - Download the ASL alphabet dataset (e.g., from Kaggle)
   - Organize it as: `data/raw/asl_alphabet_train/asl_alphabet_train/<letter>/<images>`
   - Update `config.py` if your dataset is in a different location

## Usage

### 1. Process Dataset

Extract hand landmarks from your ASL images:

```bash
python process_data.py
```

This will:
- Process all images in the dataset directory
- Extract MediaPipe hand landmarks
- Save processed data to `data/processed/asl_data.pickle`

### 2. Train Model

Train the RandomForest classifier:

```bash
python train_model.py
```

This will:
- Load processed data
- Split into train/test sets
- Train RandomForest model
- Generate learning curves
- Save model and label encoder to `models/`

### 3. Run Inference

Run real-time ASL detection from webcam:

```bash
python run_inference.py
```

Options:
- `--landmarks` or `-l`: Show hand landmarks overlay
- Press `q` to quit

### 4. Evaluate Model

Test the trained model on test dataset:

```bash
python tests/test_model.py
```

This generates:
- Accuracy score
- Classification report
- Confusion matrix
- ROC curves

### 5. Collect Custom Data

Collect your own ASL sign data:

```bash
python utils/data_collector.py [num_classes] [samples_per_class]
```

Example:
```bash
python utils/data_collector.py 26 100
```

## Configuration

Edit `config.py` to customize:

- **Dataset paths**: Update `ASL_TRAIN_DIR` and `ASL_TEST_DIR`
- **Model settings**: Adjust `N_ESTIMATORS`, `MAX_DEPTH`, etc.
- **MediaPipe settings**: Modify detection confidence and hand count
- **Image preprocessing**: Change `IMAGE_SIZE` if needed

## Technical Details

### Model Architecture
- **Algorithm**: RandomForest Classifier
- **Features**: 42 features (21 hand landmarks × 2 coordinates)
- **Preprocessing**: Hand landmark normalization relative to minimum coordinates

### Data Flow
1. **Raw Images** → MediaPipe → **Hand Landmarks** (42 features)
2. **Hand Landmarks** → RandomForest → **ASL Letter Prediction**

### MediaPipe Hand Landmarks
The model uses 21 hand landmarks:
- Wrist (1)
- Thumb (4 points)
- Index finger (4 points)
- Middle finger (4 points)
- Ring finger (4 points)
- Pinky (4 points)

Each landmark provides (x, y) coordinates normalized relative to the hand's bounding box.

## Requirements

- Python 3.7+
- opencv-python >= 4.7.0.68
- mediapipe >= 0.9.0.1
- scikit-learn >= 1.2.0
- numpy
- matplotlib
- seaborn
- tqdm

## Troubleshooting

### Camera not opening
- Check camera permissions
- Try different camera index in `run_inference.py` (change `camera_index=0` to `1` or `2`)

### Model file not found
- Ensure you've run `train_model.py` first
- Check that model files exist in `models/` directory

### Dataset path errors
- Update paths in `config.py`
- Ensure dataset is organized correctly (see Installation section)

### No hand detected
- Ensure good lighting
- Keep hand clearly visible in frame
- Adjust `MEDIAPIPE_MIN_DETECTION_CONFIDENCE` in `config.py`

## Future Improvements

- [ ] Support for words/phrases (not just letters)
- [ ] Improved model architecture (CNN, LSTM)
- [ ] Multi-hand detection
- [ ] Real-time performance optimization
- [ ] Web interface
- [ ] Mobile app support

## License

[Add your license here]

## Acknowledgments

- MediaPipe for hand tracking
- ASL alphabet dataset providers
- scikit-learn for machine learning tools
