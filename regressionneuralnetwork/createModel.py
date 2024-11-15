import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Function to convert PI/7 etc... to the actual numerical value
def angle_converter(s):
    s = s.strip()
    s = s.replace('PI', str(np.pi))
    try:
        return float(eval(s))
    except:
        raise ValueError(f"Could not convert angle '{s}' to float.")
    
# Step 1: Load Data from CSV 
# [sensor1, sensor2, ..., sensor7, distance, left start angle, right end angle]
converters = {8: angle_converter, 9: angle_converter} # Only convert the angle columns
data = np.loadtxt("test_multi_object.csv", delimiter=',', converters=converters)

# Step 2: Separate the Data
sensor_data = data[:, :7]
polar_coordinates = data[:, 7:]

# Normalize the sensor data by dividing by
# the largest recorded sensor value
max_value = np.max(sensor_data)
sensor_data /= max_value
        
# Step 3: Define Model
model = models.Sequential([
    layers.Input(shape=(7,)),            # Input layer for 7 IR sensor readings
    layers.Dense(64, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(3)                      # Output layer: 3 neurons for distance, start_angle, and end_angle
])

# Step 4: Compile the Model
model.compile(optimizer='adam', 
              loss='mean_squared_error',
              metrics=['mae'])

# Step 5: Train the Model
# Early stopping to avoid overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(sensor_data, polar_coordinates, 
                    epochs=500,
                    batch_size=32,
                    validation_split=0.2,
                    callbacks=[early_stopping])

# Step 6: Save the Model
model.save('regressionNeuralNetwork.keras')
print("Model saved as 'regressionNeuralNetwork.keras'")

# Step 7: Model Prediction
test_input = np.array([[4, 211, 18, 20, 19, 9, 8]]) / max_value

predicted_output = model.predict(test_input)
predicted_distance, predicted_start_angle, predicted_end_angle = predicted_output[0]
predicted_start_angle_deg = np.degrees(predicted_start_angle)
predicted_end_angle_deg = np.degrees(predicted_end_angle)

# Print predictions
print(f"Predicted Distance: {predicted_distance}")
print(f"Start Angle: {predicted_start_angle_deg} degrees")
print(f"End Angle: {predicted_end_angle_deg} degrees")

