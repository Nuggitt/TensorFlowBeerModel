import tensorflow as tf
from tensorflow import keras
from keras import layers, models

def create_model():
    model = models.Sequential([
        layers.InputLayer(input_shape=(224, 224, 3)),  # Input size for images
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        # Output layer with 4 units (for 4 classes) and softmax activation
        layers.Dense(6, activation='softmax')  
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model