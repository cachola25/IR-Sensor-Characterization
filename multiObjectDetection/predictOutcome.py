import asyncio
import time
import numpy as np
import tensorflow as tf

# Measure total script execution time
script_start_time = time.time()

# Step 1: Load the trained model for object prediction
load_start_time = time.time()
model = tf.keras.models.load_model('objectDetectionModel.keras')
load_end_time = time.time()

print(f"Model loaded in {load_end_time - load_start_time:.4f} seconds.")

# Step 2: Load and preprocess the data
# Example sensor data for prediction
test_input = np.array([[11,0,4,0,8,0,3]], dtype=float)
data = np.loadtxt("test.csv", delimiter=',')
sensor_data = data[:, :7]
max_value = np.max(sensor_data)
test_input /= max_value  # Normalize the input data

# Step 3: Perform model prediction
predict_start_time = time.time()
predicted_objects = model.predict(test_input)[0][0]
predict_end_time = time.time()

# Print prediction result
print(f"Predicted Number of Objects: {predicted_objects}")

# Print timing details
print(f"Model prediction completed in {predict_end_time - predict_start_time:.4f} seconds.")
print(f"Total script execution time: {time.time() - script_start_time:.4f} seconds.")
