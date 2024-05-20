import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import os

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Dictionary which assigns each label an emotion
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Load JSON and create model
json_file_path = '/Users/quynhnguyen/Downloads/Emotion demo/pretrained_model.json'
with open(json_file_path, 'r') as json_file:
    loaded_model_json = json_file.read()

# Try loading the model from JSON
try:
    emotion_model = model_from_json(loaded_model_json, custom_objects={'Sequential': Sequential})
except TypeError as e:
    print(f"Error loading model: {e}")
    exit()

# Load weights into the new model
emotion_model.load_weights("/Users/quynhnguyen/Downloads/Emotion demo/model4.h5")
print("Loaded model from disk")

# Initialize image data generator with rescaling
test_data_gen = ImageDataGenerator(rescale=1./255)

# Preprocess all test images
test_generator = test_data_gen.flow_from_directory(
    'data/test',
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical',
    shuffle=False  # Ensure that the data is not shuffled
)

# Do prediction on test data
predictions = emotion_model.predict(test_generator)

# Extract true labels from the generator
true_classes = test_generator.classes

# Get the class labels
class_labels = list(test_generator.class_indices.keys())

# Predicted classes
predicted_classes = np.argmax(predictions, axis=1)

# Print the classification report
print("Classification Report:")
report = classification_report(true_classes, predicted_classes, target_names=class_labels)
print(report)

# Print the confusion matrix
print("Confusion Matrix:")
c_matrix = confusion_matrix(true_classes, predicted_classes)
print(c_matrix)
cm_display = ConfusionMatrixDisplay(confusion_matrix=c_matrix, display_labels=class_labels)
cm_display.plot(cmap=plt.cm.Blues)
plt.show()
