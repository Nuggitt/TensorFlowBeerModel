import tensorflow as tf
from tensorflow import keras
from keras._tf_keras.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
import os
import json
from keras._tf_keras.keras.utils import Sequence
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras._tf_keras.keras.utils import to_categorical

class BeerDataGenerator(Sequence):
    def __init__(self, metadata_file, batch_size=1, target_size=(224, 224), shuffle=True):
        # Load metadata from the file only once
        with open(metadata_file, 'r') as file:
            self.metadata = json.load(file)
        
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.image_keys = list(self.metadata.keys())

        # Encode labels
        self.name_encoder = LabelEncoder().fit([item['name'] for item in self.metadata.values()])
        self.style_encoder = LabelEncoder().fit([item['style'] for item in self.metadata.values()])
        print("Unique beer names classes:", self.name_encoder.classes_)
        print("Unique beer styles classes:", self.style_encoder.classes_)

        # Initialize ImageDataGenerator for augmentation
        self.datagen = ImageDataGenerator(
            rotation_range=40,  # Random rotations up to 40 degrees
            width_shift_range=0.2,  # Random horizontal shifts (20% of the image width)
            height_shift_range=0.2,  # Random vertical shifts (20% of the image height)
            shear_range=0.2,  # Shearing transformations (distorting the image)
            zoom_range=0.2,  # Random zooming
            horizontal_flip=True,  # Random horizontal flip
            fill_mode='nearest',  # Filling pixels after transformations using nearest value
            vertical_flip=True,
            channel_shift_range=30.0
)
        
        if shuffle:
            self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.image_keys) / self.batch_size))

    def __getitem__(self, index):
        batch_keys = self.image_keys[index * self.batch_size:(index + 1) * self.batch_size]
        return self.__data_generation(batch_keys)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.image_keys)

    def __data_generation(self, batch_keys):
        images = []
        name_labels, style_labels, abv_labels, volume_labels = [], [], [], []

        for key in batch_keys:
            item = self.metadata[key]
            image_path = item['image']
            
            # Handle missing images
            if not os.path.exists(image_path):
                print(f"Image path not found: {image_path}")
                continue
            
            image = load_img(image_path, target_size=self.target_size)
            image = img_to_array(image) / 255.0  # Normalize image
            image = np.expand_dims(image, axis=0)  # Add an extra dimension for augmentation

            # Apply augmentation
            augmented_image = next(self.datagen.flow(image, batch_size=1))[0]  # Get augmented image
            images.append(augmented_image)

            # Handle labels
            name_labels.append(self.name_encoder.transform([item['name']])[0])
            style_labels.append(self.style_encoder.transform([item['style']])[0])

            abv = float(item['abv']) 
            volume = float(item['volume'])
            
            # Append labels for ABV and volume
            abv_labels.append(abv)
            volume_labels.append(volume)

        # One-hot encode categorical labels
        name_labels = to_categorical(name_labels, num_classes=len(self.name_encoder.classes_))
        style_labels = to_categorical(style_labels, num_classes=len(self.style_encoder.classes_))

        # Normalize volume if needed (e.g., Min-Max scaling)
        volume_labels = np.array(volume_labels) / max(volume_labels)  # Example normalization

        return np.array(images), {
            'name_output': np.array(name_labels),
            'style_output': np.array(style_labels),
            'abv_output': np.array(abv_labels),
            'volume_output': np.array(volume_labels)
        }

# Set up data generator
def setup_data_generators(metadata_file, batch_size=1, target_size=(224, 224)):
    return BeerDataGenerator(metadata_file=metadata_file, batch_size=batch_size, target_size=target_size)






