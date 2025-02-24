# Emotion Recognition using CNN

## Overview
This repository contains code for detecting human emotions using a Convolutional Neural Network (CNN). The model is trained on grayscale facial images and can classify emotions into seven categories:

- **Angry**
- **Disgust**
- **Fear**
- **Happy**
- **Neutral**
- **Sad**
- **Surprise**

The project includes:
- A **Jupyter Notebook** for training and testing the emotion detection model.
- A **Python script** for real-time emotion recognition using OpenCV.
- A **real-time visualization** with a softmax probability display.
- A **grey translucent UI** for better readability.

---

## Dataset
The dataset used for training can be downloaded from Kaggle:
[Emotion Detection FER Dataset](https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer)

### Dataset Structure
The dataset is expected to be structured as follows:
```
images/
  train/
    angry/
    disgust/
    fear/
    happy/
    neutral/
    sad/
    surprise/
  test/
    angry/
    disgust/
    fear/
    happy/
    neutral/
    sad/
    surprise/
```
Each folder should contain grayscale images of faces labeled according to their corresponding emotion.

---

## Installation & Dependencies
Ensure you have the following dependencies installed:
```bash
pip install -r requirements.txt
```

---

## Training the Model
The training process follows these steps:
1. **Load and preprocess** the dataset.
2. **Augment data** using random flipping, rotation, and zoom.
3. **Train a CNN model** with multiple convolutional, pooling, and dropout layers.
4. **Save the trained model** and weights (`emotiondetector.h5`).
5. **Evaluate the model** on test images.

Run the Jupyter Notebook to train the model:
```bash
jupyter notebook emotion_detection.ipynb
```

---

## Real-time Emotion Detection
To use the trained model for real-time emotion recognition via webcam:
1. Ensure a webcam is connected.
2. Run the Python script:
```bash
python real_time_emotion.py
```
3. The script detects faces, processes them, and predicts the emotion in real-time.
4. **Softmax probabilities** for all emotions are displayed dynamically on a **grey translucent panel** on the left side of the camera feed.
5. The **top predicted emotion is highlighted in green**.
6. **Times New Roman font** is used for all displayed text.

---

## Model Architecture
The CNN model consists of:
- **4 convolutional layers** with ReLU activation
- **Max-pooling layers** for feature reduction
- **Dropout layers** to prevent overfitting
- **Fully connected layers** with softmax activation for classification
- **Real-time probability visualization** with a UI overlay

---

## Results
The trained model can effectively classify emotions from facial images. Example predictions on test images are shown in the Jupyter Notebook. The real-time interface enhances usability by providing **a clean UI with a probability breakdown**.

---

## Future Improvements
- Train on a larger and more diverse dataset
- Improve model accuracy with fine-tuning
- Deploy as a web or mobile application
- Enhance UI elements with more customization options
---


## Contributors
- **Diya Uday Nayak** 
- **Profile** : https://github.com/diyanayakE15

---

## License
This project is licensed under the **MIT License**.

