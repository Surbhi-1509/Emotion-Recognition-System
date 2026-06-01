# Emotion Recognition System

A real-time facial emotion detection system built using Convolutional Neural Networks (CNN) and OpenCV. The system identifies seven human emotions from live webcam input.

## Overview

This project uses deep learning to classify facial expressions into seven categories: Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise. It was trained on the FER-2013 dataset containing over 35,000 labeled grayscale face images.

## Tech Stack

- Python 3.11
- TensorFlow / Keras
- OpenCV
- NumPy
- Scikit-learn

## Project Structure
## Project Structure
```
Emotion-Recognition-System/
├── dataset/               # FER-2013 training and test images
├── model/                 # Saved trained model
├── train.py               # CNN model training script
├── detect.py              # Real-time webcam detection
└── requirements.txt       # Project dependencies
```

## Getting Started

### Install dependencies
```bash
pip install -r requirements.txt
```

### Download Dataset
Download the FER-2013 dataset from [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013) and place it in the `dataset/` folder.

### Train the model
```bash
python train.py
```

### Run real-time detection
```bash
python detect.py
```
Press `q` to quit the webcam window.

## Model Details

- Dataset: FER-2013 (35,000+ images, 7 classes)
- Architecture: 4 convolutional layers with BatchNormalization and Dropout
- Optimizer: Adam with learning rate scheduling
- Validation Accuracy: 63%
- Training: 50 epochs on Google Colab T4 GPU

## Key Features

- Real-time emotion detection through webcam
- Data augmentation to improve model generalization
- Dropout regularization to reduce overfitting
- Early stopping to preserve the best model weights
