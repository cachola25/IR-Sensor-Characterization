import numpy as np
import tensorflow as tf
import keras_tuner as kt
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

# Define a model-building function for Keras Tuner


def build_model(hp):
    model = keras.Sequential()
    model.add(keras.Input(shape=(7,)))  # Input layer for 7 IR sensor readings

    # Tune the number of hidden layers
    for i in range(hp.Int('num_layers', 1, 3)):
        # Tune the number of units in each layer, increment by one neuron each time
        units = hp.Int(f'units_{i}', min_value=1, max_value=512, step=1)
        model.add(layers.Dense(units=units, activation='relu'))

    # Output layer: 3 neurons for distance, start_angle, end_angle
    model.add(layers.Dense(3))

    # Tune the learning rate for the optimizer
    learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='mean_squared_error',
                  metrics=['mae'])
    return model


# Step 1: Load Data from CSV
# [sensor1, sensor2, ..., sensor7, distance, left start angle, right end angle]
data = np.loadtxt("newData.csv", delimiter=',')

# Step 2: Separate the Data
sensor_data = data[:, :7]
polar_coordinates = data[:, 7:]

# Normalize the sensor data by dividing by the largest recorded sensor value
max_value = np.max(sensor_data)
sensor_data /= max_value

# Convert angles to radians for the model (optional, depending on model requirements)
# Convert start and end angles to radians
polar_coordinates[:, 1:] = np.radians(polar_coordinates[:, 1:])

# Step 3: Define Model and and create a tuner that will find the optimal architecture for us
tuner = kt.RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=10,
    executions_per_trial=1,
    directory='my_dir',
    project_name='ir_sensor_tuning',
    overwrite=True
)

# Early stopping to avoid overfitting
early_stopping = EarlyStopping(
    monitor='val_loss', patience=10, restore_best_weights=True)

# Split the data into training and validation sets
split_index = int(0.8 * len(sensor_data))
x_train, x_val = sensor_data[:split_index], sensor_data[split_index:]
y_train, y_val = polar_coordinates[:
                                   split_index], polar_coordinates[split_index:]

print(f"Beginning tuning process...")
# Step 4: Perform hyperparameter search
tuner.search(x_train, y_train,
             epochs=100,
             validation_data=(x_val, y_val),
             callbacks=[early_stopping],
             verbose=0)

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The hyperparameter search is complete.
Optimal number of layers: {best_hps.get('num_layers')}
Optimal units per layer:""")
for i in range(best_hps.get('num_layers')):
    print(f"Layer {i+1}: {best_hps.get(f'units_{i}')} units")
print(f"Optimal learning rate: {best_hps.get('learning_rate')}\n")

# Step 5: Build the model with the optimal hyperparameters and train it
model = tuner.hypermodel.build(best_hps)

history = model.fit(x_train, y_train,
                    epochs=500,
                    validation_data=(x_val, y_val),
                    callbacks=[early_stopping],
                    verbose=0)

model.save('regressionNeuralNetwork.keras')
print("Model saved as 'regressionNeuralNetwork.keras\n'")

# Step 6: Model Prediction
test_input = np.array([[4, 16, 28, 1102, 187, 9, 13]]) / max_value

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

# Define half-circle heatmap visualization


def plot_object_highlight(predicted_distance, predicted_start_angle, predicted_end_angle, max_distance, grid_size=100):
    """
    Highlight the area where the object is located on a half-circle polar plot.
    Intensity increases as it gets closer to the object's predicted distance.

    :param predicted_distance: Distance where the object is located.
    :param predicted_start_angle: Start angle of the object (in radians).
    :param predicted_end_angle: End angle of the object (in radians).
    :param max_distance: Maximum sensor detection distance.
    :param grid_size: Resolution of the polar grid.
    """
    # Define grid for theta (0 to pi) and radii (0 to max_distance)
    theta = np.linspace(0, np.pi, grid_size)
    radii = np.linspace(0, max_distance, grid_size)
    theta_grid, radii_grid = np.meshgrid(theta, radii)

    # Initialize a grid for intensities
    intensities = np.zeros_like(theta_grid)

    # Highlight the area between start and end angles
    in_angle_range = (theta_grid >= predicted_start_angle) & (
        theta_grid <= predicted_end_angle)

    # Intensity increases as distance gets closer to the predicted distance
    distance_factor = 1 - \
        np.abs(radii_grid - predicted_distance) / max_distance
    # Ensure values are in [0, 1]
    distance_factor = np.clip(distance_factor, 0, 1)

    # Combine angle range and distance factor
    intensities[in_angle_range] = distance_factor[in_angle_range]

    # Plot the highlighted area
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(10, 6))
    heatmap = ax.pcolormesh(theta_grid, radii_grid,
                            intensities, cmap="YlGnBu", shading='auto')

    # Customize plot appearance
    ax.set_theta_zero_location("E")  # 0 degrees on the right
    ax.set_theta_direction(1)  # Clockwise direction
    ax.set_ylim(0, max_distance)  # Limit radial extent
    ax.set_title("Object Highlight Visualization", va='bottom', fontsize=16)
    cbar = plt.colorbar(heatmap, ax=ax, pad=0.1)
    cbar.set_label("Intensity", rotation=270, labelpad=20)

    # Display the plot
    plt.show()


print(predicted_start_angle)

# Test prediction
# predicted_distance = 4  # Example predicted distance
predicted_start_angle = np.radians(
    predicted_start_angle_deg)  # Convert degrees to radians
predicted_end_angle = np.radians(
    predicted_end_angle_deg)  # Convert degrees to radians
max_distance = 12  # Maximum detection distance

# Plot the object highlight
plot_object_highlight(predicted_distance, predicted_start_angle,
                      predicted_end_angle, max_distance)
