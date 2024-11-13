import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from config import get_training_data_file
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping


# function to translate from array index to cell index


def index_to_cell(index):
    columns = 'ABCDEFGHIJKLMNOPQR'
    row = index // 17 + 1
    col = index % 17
    return f"{columns[col]}{row}"


def cell_to_index(cell):
    columns = 'ABCDEFGHIJKLMNOPQR'
    col = columns.index(cell[0])  # Get column index from the letter
    # Get row index (subtract 1 for 0-based index)
    row = int(cell[1:]) - 1
    return row * 17 + col


# Step 1: Load Data from CSV
data = np.loadtxt('new_database.csv', delimiter=',')

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
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

history = model.fit(sensor_data, occupancy_data,
                    epochs=1000,
                    batch_size=32,
                    validation_split=0.2,
                    callbacks=[early_stopping],
                    )

# Step 6: Save the Model
model.save('obstacle_detection_model.keras')
print("Model saved as 'obstacle_detection_model.keras'")

# Step 7: Model Prediction
test_input = np.array([[1, 1, 1, 1, 1, 1, 1]])
predicted_output = model.predict(test_input)

# Find cells with a probability above threshold
occupied_indices = np.where(predicted_output > 0.3)[1]

# Convert array index to cell coordinate pairs
occupied_cells = [index_to_cell(idx) for idx in occupied_indices]

# # Specify cells to check
# target_cells = ["E6", "L6"]

# # Print probabilities for specified cells
# for cell in target_cells:
#     index = cell_to_index(cell)
#     probability = predicted_output[0, index]
#     print(f"{cell}, with probability: {probability}")

# print()
# Iterate through occupied cells and print the corresponding probability
for i, cell in enumerate(occupied_cells):
    probability = predicted_output[0, occupied_indices[i]]
    print(f"Object in {cell}, with probability: {probability}")

# Plot training & validation accuracy values
plt.figure(figsize=(12, 4))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy over Epochs')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss over Epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

plt.show()

# Step 1: Reshape predictions into a 9x17 grid
grid_shape = (9, 17)  # Adjust based on your grid layout
probability_grid = predicted_output.reshape(grid_shape)

# Step 2: Plot heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(probability_grid, annot=True, fmt=".2f", cmap="YlGnBu", cbar=True,
            xticklabels=list('ABCDEFGHIJKLMNOPQR'), yticklabels=range(1, 10))
plt.title("Predicted Object Presence Probability per Zone")
plt.xlabel("Grid Columns")
plt.ylabel("Grid Rows")
plt.show()
