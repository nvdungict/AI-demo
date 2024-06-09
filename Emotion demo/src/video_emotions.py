import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input
import os

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Create the model
model = Sequential([
    Input(shape=(48, 48, 1)),
    Conv2D(32, kernel_size=(3, 3), activation='relu'),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(1024, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

# Load model weights
model.load_weights('/Users/quynhnguyen/Downloads/Emotion demo/model/model1.h5')

# Prevents openCL usage and unnecessary logging messages
cv2.ocl.setUseOpenCL(False)

# Dictionary which assigns each label an emotion (alphabetical order)
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Start the video file processing
video_path = '/Users/quynhnguyen/Downloads/Emotion demo/imgs/video_zoom.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Unable to open video file.")
    exit()

facecasc = cv2.CascadeClassifier('/Users/quynhnguyen/Downloads/Emotion-detection-master/src/haarcascade_frontalface_default.xml')
if facecasc.empty():
    print("Error: Unable to load the cascade classifier.")
    exit()

frame_skip = 3  # Process every 3rd frame
frame_count = 0
trackers = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Update trackers if not the detection frame
    if frame_count % frame_skip == 0:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        trackers = []

        # Initialize trackers for each detected face
        for (x, y, w, h) in faces:
            tracker = cv2.legacy.TrackerKCF_create()  # You can use other trackers like CSRT or MOSSE
            bbox = (x, y, w, h)
            tracker.init(frame, bbox)
            trackers.append(tracker)

    else:
        for tracker in trackers:
            ok, bbox = tracker.update(frame)
            if ok:
                x, y, w, h = map(int, bbox)
                cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
                roi_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[y:y + h, x:x + w]
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
                prediction = model.predict(cropped_img, verbose=0)
                maxindex = int(np.argmax(prediction))
                cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
