from beer_model import create_model
from data_preprocessing import BeerDataGenerator
import matplotlib.pyplot as plt


# Paths and constants
metadata_file = 'beer_labels_dataset/beer_metadata.json'
batch_size = 1
target_size = (224, 224)
epochs = 50

# Setup data generators
train_generator = BeerDataGenerator(metadata_file=metadata_file, batch_size=batch_size, target_size=target_size)

# Debug: Check the length of the dataset
print(f"Number of samples in the dataset: {len(train_generator)}")

# Check if the generator yields any data
for i, data in enumerate(train_generator):
    print(f"Batch {i}: {data}")
    if i >= 2:  # Check the first 3 batches
        break

# Create the model
model = create_model()

# Check if data is available
if len(train_generator) == 0:
    raise ValueError("No data found. Please check the metadata file and image paths.")

# Train the model
history = model.fit(
    train_generator,
    epochs=epochs,
    steps_per_epoch=len(train_generator),
    verbose=1
)

# Save the model
model.save('beer_model.keras')

# Plot the metrics
plt.figure(figsize=(12, 8))

# Plot accuracy for name and style outputs
plt.subplot(2, 1, 1)
plt.plot(history.history['name_output_accuracy'], label='Name Accuracy')
plt.plot(history.history['style_output_accuracy'], label='Style Accuracy')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plot loss for all outputs
plt.subplot(2, 1, 2)
plt.plot(history.history['loss'], label='Total Loss')
plt.plot(history.history['name_output_loss'], label='Name Loss')
plt.plot(history.history['style_output_loss'], label='Style Loss')
plt.plot(history.history['abv_output_loss'], label='ABV Loss')
plt.plot(history.history['volume_output_loss'], label='Volume Loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Evaluate the model
test_results = model.evaluate(train_generator)
print(f"Evaluation Results: {test_results}")

# Unpack and print the evaluation results
test_loss, test_name_acc, test_style_acc, test_abv_mae, test_volume_mae, *extra_metrics = test_results

print(f"Test Loss: {test_loss}")
print(f"Test Name Accuracy: {test_name_acc * 100:.2f}%")
print(f"Test Style Accuracy: {test_style_acc * 100:.2f}%")
print(f"Test ABV MAE: {test_abv_mae}")
print(f"Test Volume MAE: {test_volume_mae}")
print(f"Other Metrics: {extra_metrics}")
