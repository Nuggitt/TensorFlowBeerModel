import tensorflow as tf
from tensorflow import keras
from keras import layers, models

def create_model():
    # Define the input layer
    input_layer = layers.Input(shape=(224, 224, 3))
    
    # Build the base model
    base_model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
    ])
    
    # Base model output
    base_model_output = base_model(input_layer)
    
    # Output layers for name, style, ABV, and volume
    name_output = layers.Dense(6, activation='softmax', name="name_output")(base_model_output)
    style_output = layers.Dense(6, activation='softmax', name="style_output")(base_model_output)
    abv_output = layers.Dense(1, activation='linear', name="abv_output")(base_model_output)
    volume_output = layers.Dense(1, activation='linear', name="volume_output")(base_model_output)

    # Define the model with inputs and outputs
    model = models.Model(inputs=input_layer, outputs=[name_output, style_output, abv_output, volume_output])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss={
            "name_output": "categorical_crossentropy",
            "style_output": "categorical_crossentropy",
            "abv_output": "mean_squared_error",
            "volume_output": "mean_squared_error"
        },
        metrics={
            "name_output": "accuracy",
            "style_output": "accuracy",
            "abv_output": "mae",
            "volume_output": "mae"
        }
    )
    
    return model
