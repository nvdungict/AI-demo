from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Recreate the model architecture
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

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy')

# Load the weights into the model
model.load_weights('modelmain.h5')

# Get JSON-compatible representation of model architecture
model_json = model.to_json()

# Save JSON representation to a file
with open('modelmain.json', 'w') as json_file:
    json_file.write(model_json)

# Optionally, save the weights again
model.save_weights('modelweight.h5')
