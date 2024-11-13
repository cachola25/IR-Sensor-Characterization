import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# function to translate from array index to cell index
def index_to_cell(index):
    columns = 'ABCDEFGHIJKLMNOPQR' 
    row = index // 17 + 1
    col = index % 17
    return f"{columns[col]}{row}"

def cell_to_index(cell):
    columns = 'ABCDEFGHIJKLMNOPQR' 
    col = columns.index(cell[0])  # Get column index from the letter
    row = int(cell[1:]) - 1       # Get row index (subtract 1 for 0-based index)
    return row * 17 + col

# Step 1: Load Data from CSV
data = np.loadtxt('ir_sensor_data.csv', delimiter=',')

# Step 2: Separate the Data where first 7 col are IR sensor readings and remaining is the array
sensor_data = data[:, :7]
occupancy_data = data[:, 7:]

# Step 3: Define the Model (input layer of 7, first layer of 64, second layer of 128, and output layer of 153)
model = models.Sequential([
    layers.Input(shape=(7,)),
    layers.Dense(len(sensor_data), activation='relu'), 
    layers.Dense(len(sensor_data) // 2, activation='relu'),
    layers.Dense(len(occupancy_data[0]), activation='sigmoid')
])

# Step 4: Compile the Model (binary_crossentropy for multi label classification)
model.compile(optimizer='adam', 
              loss='binary_crossentropy',
              metrics=['accuracy'])

tf.random.set_seed(42)
# Step 5: Train the Model (20 epochs, 32 samples per gradient update, and 80/20 split for training/testing)

history = model.fit(sensor_data, occupancy_data, 
                    epochs=20,
                    batch_size=32,
                    validation_split=0.2,
                    )

# Step 6: Save the Model
model.save('obstacle_detection_model.h5')
print("Model saved as 'obstacle_detection_model.h5'")

# Step 7: Model Prediction
test_input = np.array([[5,201,19,14,54,13,12]])
predicted_output = model.predict(test_input)

# Find cells with a probability above threshold
occupied_indices = np.where(predicted_output > 0.1)[1]

# Convert array index to cell coordinate pairs
occupied_cells = [index_to_cell(idx) for idx in occupied_indices]

# Specify cells to check
target_cells = ["E6", "K4"]

# Print probabilities for specified cells
for cell in target_cells:
    index = cell_to_index(cell)
    probability = predicted_output[0, index]
    print(f"{cell}, with probability: {probability}")

print()

# Iterate through occupied cells and print the corresponding probability
for i, cell in enumerate(occupied_cells):
    probability = predicted_output[0, occupied_indices[i]]
    print(f"Object in {cell}, with probability: {probability}")

# Step 7: Model Prediction
test_input = np.array([[2,204,14,16,56,11,11]])
predicted_output = model.predict(test_input)

# Reshape the output into a 9x17 grid
probability_grid = predicted_output.reshape(9, 17)

# Print the grid of probabilities with labels for better readability
print("Predicted Probability Grid (9x17):")
columns = 'ABCDEFGHIJKLMNOPQR'
print("    " + " ".join(columns))  # Print column headers
for i, row in enumerate(probability_grid):
    row_str = " ".join(f"{prob:.2f}" for prob in row)  # Format each probability to two decimal places
    print(f"{i+1:2} | {row_str}")



