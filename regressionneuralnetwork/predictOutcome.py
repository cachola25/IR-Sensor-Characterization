#
# Integrated script to control the robot based on predictions from both models.
#
import asyncio
import time
import numpy as np
import tensorflow as tf

# Load the trained models

start_time = time.time()
model = tf.keras.models.load_model('differentSizes2.keras')


data = np.loadtxt("differentSizesData/testDifferentSizesData.csv", delimiter=',')
sensor_data = data[:, :7]
polar_coordinates = data[:, 7:]
max_value = np.max(sensor_data)

# Step 6: Model Prediction
test_input = np.array([[144,17,24,23,19,9,15]]) / max_value

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
diagram_inputs = (sensor_data / max_value)

# Ground truth for comparison (distance, start_angle, end_angle in degrees)
ground_truth = polar_coordinates

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
end_time = time.time()

print(f"Execution Time: {end_time - start_time} seconds")
