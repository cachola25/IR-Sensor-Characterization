import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Step 1: Load Data from CSV 
# [sensor1, sensor2, ..., sensor7, distance, left start angle, right end angle]
data = np.loadtxt('b.csv', delimiter=',')

# Step 2: Separate the Data
sensor_data = data[:, :7]
polar_coordinates = data[:, 7:]

# Step 3: Define Model
model = models.Sequential([
    layers.Input(shape=(7,)),            # Input layer for 7 IR sensor readings
    layers.Dense(64, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(3)                      # Output layer: 3 neurons for distance, start_angle, and end_angle
])

# Step 4: Compile the Model
# Use mean squared error as we are predicting continuous values
model.compile(optimizer='adam', 
              loss='mean_squared_error',
              metrics=['mae'])           # Mean Absolute Error for easier interpretation of error

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
test_input = np.array([[4,211,18,20,19,9,8]])
predicted_output = model.predict(test_input)

# Print prediction
predicted_distance, predicted_start_angle, predicted_end_angle = predicted_output[0]
print(f"Predicted Distance: {predicted_distance}, Start Angle: {predicted_start_angle}, End Angle: {predicted_end_angle}")

