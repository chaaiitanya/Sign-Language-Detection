# Camera Setup Instructions

## MediaPipe Update Complete ✓

The MediaPipe 0.10+ API has been successfully integrated:
- ✓ Hand landmarker model downloaded (7.5MB)
- ✓ MediaPipe Hand Landmarker initialized
- ✓ ASL detector ready

## Camera Access Issue

If you see "Failed to grab frame", follow these steps:

### macOS Camera Permissions

1. **Grant Camera Permission:**
   - Open **System Preferences** → **Security & Privacy** → **Privacy** → **Camera**
   - Make sure **Terminal** (or your Python IDE) is checked
   - If not listed, you may need to run the app once to trigger the permission request

2. **Reset Camera Permissions (if needed):**
   ```bash
   tccutil reset Camera
   ```
   Then restart your terminal/IDE and try again.

3. **Check Camera Availability:**
   ```bash
   python3 -c "import cv2; cap = cv2.VideoCapture(0); print('Camera available:', cap.isOpened()); cap.release()"
   ```

### Running the Project

Once camera permissions are granted:

```bash
# Basic detection
python3 run_inference.py

# With hand landmarks visualization
python3 run_inference.py --landmarks
```

### Troubleshooting

- **"Could not open camera"**: Check camera permissions
- **"Failed to grab frame"**: Camera may be in use by another app
- **No hand detected**: Make sure your hand is clearly visible in good lighting

### What's Working

✓ Model trained (98.93% accuracy)
✓ MediaPipe 0.10+ API integrated
✓ Hand landmarker model loaded
✓ Inference code ready
✓ All dependencies installed

The project is ready - just needs camera access!
