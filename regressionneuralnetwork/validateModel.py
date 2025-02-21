import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

# Load the trained model
model = keras.models.load_model('differentSizes.keras')

# Load the dataset
file_path_trained = "differentSizesData/testDifferentSizesDataShuffle.csv"
file_path = "validation.csv"

data = np.loadtxt(file_path, delimiter=',')
data_trained = np.loadtxt(file_path_trained, delimiter=',')

# Separate sensor data and polar coordinates
sensor_data_trained = data_trained[:, :7]
sensor_data = data[:, :7]
polar_coordinates = data[:, 7:]


# Normalize the sensor data using the same max_value as in training
max_value = np.max(sensor_data_trained)
sensor_data /= max_value

# Convert angles from degrees to radians for comparison
polar_coordinates[:, 1:] = np.radians(polar_coordinates[:, 1:])

# Split into training and validation sets
split_index = int(0.0 * len(sensor_data))
x_val = sensor_data[split_index:]
y_val = polar_coordinates[split_index:]

# Perform predictions
predictions = model.predict(x_val)

# Convert predictions and ground truth angles back to degrees
true_distances = y_val[:, 0]
true_start_angles = np.degrees(y_val[:, 1])
true_end_angles = np.degrees(y_val[:, 2])

predicted_distances = predictions[:, 0]
predicted_start_angles = np.degrees(predictions[:, 1])
predicted_end_angles = np.degrees(predictions[:, 2])

# Compute absolute errors
distance_errors = np.abs(predicted_distances - true_distances)
start_angle_errors = np.abs(predicted_start_angles - true_start_angles)
end_angle_errors = np.abs(predicted_end_angles - true_end_angles)

# Compute average errors
avg_distance_error = np.mean(distance_errors)
avg_start_angle_error = np.mean(start_angle_errors)
avg_end_angle_error = np.mean(end_angle_errors)

# Print results
print(f"Average Distance Error: {avg_distance_error} inches")
print(f"Average Start Angle Error: {avg_start_angle_error} degrees")
print(f"Average End Angle Error: {avg_end_angle_error} degrees")

# Identify the top 10 highest error points for each metric
top_10_distance_indices = np.argsort(distance_errors)[-10:][::-1]
top_10_start_angle_indices = np.argsort(start_angle_errors)[-10:][::-1]
top_10_end_angle_indices = np.argsort(end_angle_errors)[-10:][::-1]

print("\nTop 10 highest distance error points:")
for i in top_10_distance_indices:
    print(f"True Distance: {true_distances[i]}, Predicted Distance: {predicted_distances[i]}, Error: {distance_errors[i]}")

print("\nTop 10 highest start angle error points:")
for i in top_10_start_angle_indices:
    print(f"True Start Angle: {true_start_angles[i]} degrees, Predicted Start Angle: {predicted_start_angles[i]} degrees, Error: {start_angle_errors[i]}")

print("\nTop 10 highest end angle error points:")
for i in top_10_end_angle_indices:
    print(f"True End Angle: {true_end_angles[i]} degrees, Predicted End Angle: {predicted_end_angles[i]} degrees, Error: {end_angle_errors[i]}")

