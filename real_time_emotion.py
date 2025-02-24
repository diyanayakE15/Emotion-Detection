import cv2
import numpy as np
from keras.models import model_from_json

# Load the pre-trained model
json_file = open("C:\\Users\\Diya Nayak\\Desktop\\ML_Projects\\EmotionRecognition\\emotiondetector.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("C:\\Users\\Diya Nayak\\Desktop\\ML_Projects\\EmotionRecognition\\emotiondetector.h5")

# Load Haarcascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Emotion labels
labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

# Function to preprocess input image
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0  # Normalize

# Open webcam
webcam = cv2.VideoCapture(0)

while True:
    ret, frame = webcam.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        img = extract_features(face)
        
        # Get predictions
        pred = model.predict(img)[0]
        predicted_label = labels[np.argmax(pred)]
        
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Display predicted emotion
        cv2.putText(frame, f"Emotion: {predicted_label}", (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2)

    # Draw a grey translucent box for softmax probabilities
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (200, 250), (100, 100, 100), -1)  # Grey box
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)  # Adjust transparency
    
    # Display softmax probabilities on the left side
    start_y = 30  # Starting position for text
    for i, (emotion, probability) in enumerate(zip(labels.values(), pred)):
        text = f"{emotion}: {probability:.2f}"
        color = (0, 255, 0) if emotion == predicted_label else (255, 255, 255)  # Highlight top emotion
        
        # Using Times New Roman Font
        cv2.putText(frame, text, (10, start_y + i * 30), cv2.FONT_HERSHEY_TRIPLEX, 0.6, color, 1)

    # Show the frame
    cv2.imshow("Real-Time Emotion Recognition", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

webcam.release()
cv2.destroyAllWindows()
