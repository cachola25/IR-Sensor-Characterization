import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

# Load the trained model
model = keras.models.load_model('outcomePredictionModel3.keras')

# Load the dataset
file_path = "test3Shuffled.csv"
data = np.loadtxt(file_path, delimiter=',')

# Separate sensor data and ground truth labels
sensor_data = data[:, :7]
outcome_labels = data[:, 7].astype(int)

# Normalize the sensor data using the same max_value as in training
max_value = np.max(sensor_data)
sensor_data /= max_value

# Split into training and validation sets
split_index = int(0.8 * len(sensor_data))
x_val = sensor_data[split_index:]
y_val = outcome_labels[split_index:]

# Perform predictions
predictions = model.predict(x_val)
predicted_classes = np.argmax(predictions, axis=1)

# Calculate correctness
correct_predictions = np.sum(predicted_classes == y_val)
incorrect_predictions = len(predicted_classes) - correct_predictions

# Print results
print(f"Correct Predictions: {correct_predictions}")
print(f"Incorrect Predictions: {incorrect_predictions}")
print(f"Accuracy: {correct_predictions / len(predicted_classes) * 100:.2f}%")
