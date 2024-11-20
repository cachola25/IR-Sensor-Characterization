#
# Integrated script to control the robot based on predictions from both models.
#
import asyncio
import numpy as np
import tensorflow as tf

from irobot_edu_sdk.robots import Create3, event
from irobot_edu_sdk.backend.bluetooth import Bluetooth

robot_name = "Roomba"
robot = Create3(Bluetooth(robot_name))
# Load the trained models
model = tf.keras.models.load_model('regressionNeuralNetworkNoZeros.keras')

data = np.loadtxt("finalNewDataNoZeros.csv", delimiter=',')
sensor_data = data[:, :7]
polar_coordinates = data[:, 7:]
max_value = np.max(sensor_data)

# Step 6: Model Prediction
test_input = np.array([[0, 0, 2, 1, 4, 0, 8]]) / max_value

predicted_output = model.predict(test_input)
predicted_distance, predicted_start_angle, predicted_end_angle = predicted_output[0]
predicted_start_angle_deg = np.degrees(predicted_start_angle)
predicted_end_angle_deg = np.degrees(predicted_end_angle)

if predicted_start_angle_deg > 180:
    predicted_start_angle_deg -= 180
if predicted_end_angle_deg > 180:
    predicted_end_angle_deg -= 180
# Print predictions
print(f"Predicted Distance: {predicted_distance}")
print(f"Start Angle: {predicted_start_angle_deg} degrees")
print(f"End Angle: {predicted_end_angle_deg} degrees")

# ----------------------------------------------#
# The array of test inputs to calculate error from each bin
diagram_inputs = np.array([
    [2, 0, 2, 3, 8, 0, 972],
    [0, 0, 3, 6, 7, 0, 274],
    [0, 1, 2, 5, 6, 0, 80],
    [0, 1, 0, 4, 6, 0, 17],
    [0, 0, 1, 3, 4, 0, 5],
    [0, 1, 3, 6, 6, 2, 4],
    [0, 1, 0, 6, 5, 652, 0],
    [0, 0, 3, 2, 8, 151, 3],
    [0, 1, 0, 6, 2, 38, 0],
    [0, 0, 0, 4, 7, 10, 1],
    [0, 2, 2, 1, 7, 2, 4],
    [0, 0, 1, 4, 6, 2, 2],
    [0, 0, 4, 6, 117, 18, 3],
    [0, 0, 0, 3, 24, 1, 0],
    [0, 1, 2, 6, 10, 3, 0],
    [0, 0, 4, 3, 10, 0, 4],
    [0, 0, 4, 6, 7, 2, 3],
    [0, 0, 1, 2, 7, 0, 1],
    [0, 0, 7, 1125, 163, 1, 0],
    [0, 0, 69, 268, 81, 1, 0],
    [1, 2, 104, 95, 42, 14, 5],
    [0, 1, 55, 36, 22, 12, 2],
    [1, 3, 24, 17, 14, 7, 4],
    [0, 0, 8, 10, 6, 4, 0],
    [0, 5, 904, 28, 9, 2, 4],
    [1, 2, 139, 17, 7, 1, 4],
    [0, 1, 28, 19, 7, 1, 3],
    [1, 0, 11, 14, 8, 0, 4],
    [1, 2, 6, 16, 6, 1, 3],
    [2, 0, 4, 10, 7, 0, 3],
    [0, 998, 0, 10, 0, 6, 0],
    [0, 191, 6, 10, 7, 0, 4],
    [0, 56, 6, 22, 5, 3, 0],
    [0, 18, 3, 15, 3, 0, 1],
    [0, 6, 4, 10, 6, 0, 3],
    [0, 7, 3, 11, 6, 4, 1],
    [584, 14, 4, 5, 8, 0, 2],
    [139, 5, 4, 4, 8, 0, 4],
    [37, 5, 3, 6, 6, 0, 3],
    [9, 4, 3, 5, 7, 1, 2],
    [7, 0, 5, 1, 7, 0, 3],
    [3, 1, 6, 5, 11, 0, 4]
])

# Ground truth for comparison (distance, start_angle, end_angle in degrees)
ground_truth = np.array([
    [2.0, 39.0, 21.0],
    [4.0, 38.0, 22.0],
    [6.0, 37.0, 23.0],
    [8.0, 36.0, 24.0],
    [10.0, 35.0, 25.0],
    [12.0, 34.0, 26.0],
    [2.0, 59.0, 41.0],
    [4.0, 58.0, 42.0],
    [6.0, 57.0, 43.0],
    [8.0, 56.0, 44.0],
    [10.0, 55.0, 45.0],
    [12.0, 54.0, 46.0],
    [2.0, 79.0, 61.0],
    [4.0, 78.0, 62.0],
    [6.0, 77.0, 63.0],
    [8.0, 76.0, 64.0],
    [10.0, 75.0, 65.0],
    [12.0, 74.0, 66.0],
    [2.0, 99.0, 81.0],
    [4.0, 98.0, 82.0],
    [6.0, 97.0, 83.0],
    [8.0, 96.0, 84.0],
    [10.0, 95.0, 85.0],
    [12.0, 94.0, 86.0],
    [2.0, 119.0, 101.0],
    [4.0, 118.0, 102.0],
    [6.0, 117.0, 103.0],
    [8.0, 116.0, 104.0],
    [10.0, 115.0, 105.0],
    [12.0, 114.0, 106.0],
    [2.0, 139.0, 121.0],
    [4.0, 138.0, 122.0],
    [6.0, 137.0, 123.0],
    [8.0, 136.0, 124.0],
    [10.0, 135.0, 125.0],
    [12.0, 134.0, 126.0],
    [2.0, 159.0, 141.0],
    [4.0, 158.0, 142.0],
    [6.0, 157.0, 143.0],
    [8.0, 156.0, 144.0],
    [10.0, 155.0, 145.0],
    [12.0, 154.0, 146.0]
])

# Normalize the the diagram inputs based on the max value from training
max_value = np.max(diagram_inputs)
diagram_inputs = diagram_inputs.astype(float) / max_value

# predict outcomes
predictions = model.predict(diagram_inputs)

# convert angles to degrees for easier comparison
predictions[:, 1:] = np.degrees(predictions[:, 1:])

# return predictions and ground truth
np.save("predictions.npy", predictions)
np.save("ground_truth.npy", ground_truth)
