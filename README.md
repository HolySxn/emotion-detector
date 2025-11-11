# Face Emotion Recognition App 🎭

A Streamlit web application for detecting and classifying facial emotions using a trained CNN model.

## Features

- 📸 **Image Upload**: Upload photos to detect emotions from faces
- 📷 **Camera Input**: Use your webcam to capture photos and analyze emotions in real-time
- 🎯 **Multi-Face Detection**: Detects and analyzes multiple faces in a single image
- 📊 **Confidence Scores**: Shows probability distribution for all emotion classes
- 🎨 **Visual Feedback**: Annotated images with bounding boxes and emotion labels

## Detected Emotions

The model can recognize the following emotions:

- 😠 Angry
- 😨 Fear
- 😊 Happy
- 😢 Sad
- 😮 Surprise

## Installation

1. **Clone or navigate to the project directory**

2. **Install dependencies**

   For the simple version (recommended):

   ```bash
   pip install -r requirements_simple.txt
   ```

   For the full version with real-time video streaming:

   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure the trained model is present**

   Make sure `emotion_cnn_full.pth` is in the same directory as the app files.

## Usage

### Option 1: Simple App (Recommended)

Run the simplified version with image upload and camera capture:

```bash
streamlit run streamlit_app.py
```

### Option 2: Full App (With WebRTC streaming)

Run the full version with real-time video streaming:

```bash
streamlit run app.py
```

The app will open in your default web browser at `http://localhost:8501`

## How to Use

1. **Upload Image Tab**:

   - Click "Browse files" to upload an image (JPG, JPEG, or PNG)
   - The app will detect faces and display emotions with confidence scores
   - View detailed probability distributions for each detected face

2. **Camera Input Tab**:
   - Click "Take a picture" to capture a photo using your webcam
   - Allow camera access when prompted by your browser
   - The app will analyze the captured image and show detected emotions

## Model Information

- **Architecture**: Custom CNN with 3 convolutional blocks and 3 fully connected layers
- **Input Size**: 128x128 RGB images
- **Normalization**: Mean=[0.511, 0.509, 0.508], Std=[0.251, 0.250, 0.250]
- **Number of Classes**: 5 emotions

## Technical Details

### Model Architecture

```
- Conv Block 1: Conv2d(3→32) + BatchNorm + ReLU + MaxPool + Dropout(0.15)
- Conv Block 2: Conv2d(32→64) + BatchNorm + ReLU + MaxPool + Dropout(0.15)
- Conv Block 3: Conv2d(64→128) + BatchNorm + ReLU + MaxPool + Dropout(0.20)
- FC Block 1: Linear(32768→256) + BatchNorm + ReLU + Dropout(0.30)
- FC Block 2: Linear(256→128) + BatchNorm + ReLU + Dropout(0.30)
- Output: Linear(128→5)
```

### Face Detection

The app uses OpenCV's Haar Cascade classifier for face detection before emotion classification.

## Troubleshooting

### Model not found

```
Error loading model: [Errno 2] No such file or directory: 'emotion_cnn_full.pth'
```

**Solution**: Ensure `emotion_cnn_full.pth` is in the same directory as the app file. You can generate it by running the last cell in `final.ipynb`.

### Camera not working

**Solution**:

- Make sure you've allowed camera access in your browser
- Try refreshing the page
- Check if another application is using the camera

### CUDA/GPU issues

The app automatically detects if CUDA is available. If you encounter GPU-related errors:

```bash
# The model will automatically fall back to CPU
```

## Requirements

- Python 3.8+
- See `requirements_simple.txt` or `requirements.txt` for package dependencies

## Project Structure

```
.
├── app.py                      # Full version with WebRTC
├── streamlit_app.py            # Simplified version (recommended)
├── emotion_cnn.pth             # Model weights only
├── emotion_cnn_full.pth        # Full model (architecture + weights)
├── requirements.txt            # Dependencies (full version)
├── requirements_simple.txt     # Dependencies (simple version)
├── README.md                   # This file
└── final.ipynb                 # Training notebook
```

## Notes

- The simple version (`streamlit_app.py`) is recommended for most users as it's easier to set up
- The full version (`app.py`) includes WebRTC for true real-time video streaming but requires additional dependencies
- Both apps now load the full model (`emotion_cnn_full.pth`) which contains both architecture and weights
- Face detection works best with clear, front-facing photos
- Lighting conditions can affect emotion detection accuracy

## License

This project is for educational purposes.

## Credits

Built with:

- [Streamlit](https://streamlit.io/)
- [PyTorch](https://pytorch.org/)
- [OpenCV](https://opencv.org/)
