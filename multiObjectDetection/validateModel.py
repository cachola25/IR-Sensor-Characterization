import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

# Load the trained model
model = keras.models.load_model('objectDetectionModel.keras')

# Load the dataset
file_path = "testShuffled.csv"
data = np.loadtxt(file_path, delimiter=',')

# Separate sensor data and ground truth labels
sensor_data = data[:, :7]
num_objects = data[:, 7]

# Normalize the sensor data using the same max_value as in training
max_value = np.max(sensor_data)
sensor_data /= max_value

# Split into training and validation sets
split_index = int(0.8 * len(sensor_data))
x_val = sensor_data[split_index:]
y_val = num_objects[split_index:]

# Perform predictions
predictions = model.predict(x_val)

# Apply cutoff threshold to determine final predictions
cutoff = 0.5
final_predictions = (predictions.flatten() > cutoff).astype(int)
true_labels = y_val.astype(int)

# Calculate correctness
correct_predictions = np.sum(final_predictions == true_labels)
incorrect_predictions = len(final_predictions) - correct_predictions

# Print results
print(f"Correct Predictions: {correct_predictions}")
print(f"Incorrect Predictions: {incorrect_predictions}")
print(f"Accuracy: {correct_predictions / len(final_predictions) * 100:.2f}%")
