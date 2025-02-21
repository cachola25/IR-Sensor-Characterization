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
test_input = np.array([[14,14,13,22,32,10,113]]) / max_value

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

end_time = time.time()

print(f"Execution Time: {end_time - start_time} seconds")