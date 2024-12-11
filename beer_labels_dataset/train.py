from beer_model import create_model
from data_preprocessing import setup_data_generators
import matplotlib.pyplot as plt

train_dir = 'beer_labels_dataset/train/'
validation_dir = 'beer_labels_dataset/validation/'
test_dir = 'beer_labels_dataset/test/'

# Setup the data generators
train_generator, validation_generator, test_generator = setup_data_generators(train_dir, validation_dir, test_dir)

# Create the model
model = create_model()

# Train the model without callbacks and with fewer epochs
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=30,  # Fewer epochs for simplicity
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size
)

model.save('beer_model.keras')

# Plot the training and validation accuracy and loss
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()


# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples // test_generator.batch_size)
print(f"Test accuracy: {test_acc * 100:.2f}%")
