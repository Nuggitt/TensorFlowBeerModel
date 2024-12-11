import tensorflow as tf

def convert_model_to_tflite(model_path, tflite_model_path):
    # Load the saved model
    model = tf.keras.models.load_model(model_path)

    # Convert the model to TensorFlow Lite format
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save the TensorFlow Lite model
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)
    print(f"Model converted to TensorFlow Lite format and saved as {tflite_model_path}")

# Convert the saved model to TFLite format
convert_model_to_tflite('beer_model.keras', 'beer_model.tflite')
