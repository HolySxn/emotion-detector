# Face Emotion Recognition App 🎭

A Streamlit web application for detecting and classifying facial emotions using trained CNN models. The app supports both image upload and real-time camera input with face detection and emotion classification.

## Features

- 📸 **Image Upload**: Upload photos to detect emotions from faces
- 📷 **Camera Input**: Use your webcam to capture photos and analyze emotions in real-time
- 🎥 **WebRTC Streaming**: Real-time video emotion detection
- 🎯 **Multi-Face Detection**: Detects and analyzes multiple faces in a single image
- 📊 **Confidence Scores**: Shows probability distribution for all emotion classes
- 🎨 **Visual Feedback**: Annotated images with bounding boxes and emotion labels
- 🔧 **Flexible Model Loading**: Specify custom model path via command-line argument

## Detected Emotions

The models can recognize the following 5 emotions:

- 😠 **Angry**
- 😨 **Fear**
- 😊 **Happy**
- 😢 **Sad**
- 😮 **Surprise**

## Models

- `emotion_cnn.pth`: Original model trained on the base dataset.
- `emotion_cnn_aug.pth`: Model trained with augmented data for improved robustness.

## Training Notebooks

- `final.ipynb`: Original training notebook.
- `final_aug_training.ipynb`: Notebook for training with augmented data.
- `augmentation.ipynb`: Notebook demonstrating data augmentation techniques.

## Installation

### Prerequisites

- Python 3.8+
- Virtual environment (recommended)

### Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/HolySxn/emotion-detector.git
   cd emotion-detector
   ```

2. **Create and activate virtual environment**

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Linux/Mac
   # or
   .venv\Scripts\activate  # On Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure the trained models are present**

   Make sure `emotion_cnn.pth` and `emotion_cnn_aug.pth` are in the `models/` directory. If you don't have them:

   - Train the models using the provided notebooks.
   - Or download pre-trained weights (if available).

## Usage

### Basic Usage

Run the app with the default model (`emotion_cnn.pth`):

```bash
streamlit run app.py
```

### Advanced Usage

**Specify a custom model file:**

```bash
streamlit run app.py -- --model models/emotion_cnn_aug.pth
```

**Get help:**

```bash
streamlit run app.py -- --help
```

### Command-Line Arguments

- `--model`: Path to the model file (default: `emotion_cnn.pth`)

**Note:** Use `--` to separate Streamlit arguments from app arguments.

The app will automatically open in your default web browser at `http://localhost:8501`

## Project Structure

```
.
├── app.py                      # Main Streamlit application
├── model.py                    # CNN model definition
├── models/                     # Directory for trained models
│   ├── emotion_cnn.pth         # Base model
│   └── emotion_cnn_aug.pth     # Augmented model
├── training/                   # Training notebooks
│   ├── final.ipynb             # Original training notebook
│   ├── final_aug_training.ipynb# Augmented training notebook
│   └── augmentation.ipynb      # Data augmentation notebook
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── MODEL.md                    # Model description
├── .gitignore                  # Git ignore rules
└── .venv/                      # Virtual environment (not in repo)
```

## Acknowledgments

Built with:

- [Streamlit](https://streamlit.io/) - Web framework
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [OpenCV](https://opencv.org/) - Computer vision library
- [streamlit-webrtc](https://github.com/whitphx/streamlit-webrtc) - Real-time video streaming
