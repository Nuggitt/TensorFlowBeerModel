from tensorflow import keras
from keras._tf_keras.keras.preprocessing.image import load_img, img_to_array
import os

image_path = "beer_labels_dataset/train/Tuborg/tuborg-classic.jpg"  # Replace with an image from your dataset
try:
    img = load_img(image_path)
    img_array = img_to_array(img)
    print(f"Loaded image shape: {img_array.shape}")
except Exception as e:
    print(f"Error loading image: {e}")