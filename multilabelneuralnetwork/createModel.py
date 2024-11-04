import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# function to translate from array index to cell index
def index_to_cell(index):
    columns = 'ABCDEFGHIJKLMNOPQR' 
    row = index // 17 + 1
    col = index % 17
    return f"{columns[col]}{row}"

# Step 1: Load Data from CSV
data = np.loadtxt('ir_sensor_data.csv', delimiter=',')

# Step 2: Separate the Data where first 7 col are IR sensor readings and remaining is the array
sensor_data = data[:, :7]
occupancy_data = data[:, 7:]

# Step 3: Define the Model (input layer of 7, first layer of 64, second layer of 128, and output layer of 153)
model = models.Sequential([
    layers.Input(shape=(7,)),
    layers.Dense(64, activation='relu'), 
    layers.Dense(128, activation='relu'),
    layers.Dense(153, activation='sigmoid')
])

# Step 4: Compile the Model (binary_crossentropy for multi label classification)
model.compile(optimizer='adam', 
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Step 5: Train the Model (20 epochs, 32 samples per gradient update, and 80/20 split for training/testing)
history = model.fit(sensor_data, occupancy_data, 
                    epochs=20,
                    batch_size=32,
                    validation_split=0.2)

# Step 6: Save the Model
model.save('obstacle_detection_model.h5')
print("Model saved as 'obstacle_detection_model.h5'")

# Step 7: Model Prediction
test_input = np.array([[3, 206, 15, 10, 55, 12, 11]])
predicted_output = model.predict(test_input)

# Find cells with a probability above threshold
occupied_indices = np.where(predicted_output > 0.5)[1]

# Convert array index to cell coordinate pairs
occupied_cells = [index_to_cell(idx) for idx in occupied_indices]
print("Predicted occupied cells:", occupied_cells)



