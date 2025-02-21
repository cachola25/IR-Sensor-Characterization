import asyncio
import time
import numpy as np
import tensorflow as tf

# Measure total script execution time
script_start_time = time.time()

# Load the trained model for object prediction
load_start_time = time.time()
model = tf.keras.models.load_model('outcomePredictionModel3.keras')
load_end_time = time.time()

print(f"Model loaded in {load_end_time - load_start_time:.4f} seconds.")

# Load and preprocess the data
test_input = np.array([[47, 74, 12, 27, 14, 7, 82]], dtype=float)
data = np.loadtxt("test3.csv", delimiter=',')
sensor_data = data[:, :7]
max_value = np.max(sensor_data)
test_input /= max_value 

# Perform model prediction
predict_start_time = time.time()
predicted_probabilities = model.predict(test_input)[0]
predicted_class = np.argmax(predicted_probabilities)
predict_end_time = time.time()

# Print prediction result
print(f"Predicted Number of Objects: {predicted_class}")
print(f"Predicted Probabilites: {predicted_probabilities}")


# Print timing details
print(f"Model prediction completed in {predict_end_time - predict_start_time:.4f} seconds.")
print(f"Total script execution time: {time.time() - script_start_time:.4f} seconds.")
