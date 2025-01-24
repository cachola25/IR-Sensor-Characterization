import numpy as np
import tensorflow as tf
import keras_tuner as kt
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

# load predictions and ground truth
predictions = np.load("predictions.npy")
ground_truth = np.load("ground_truth.npy")

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
                  metrics=['mse'])
    return model


# Step 1: Load Data from CSV
# [sensor1, sensor2, ..., sensor7, distance, left start angle, right end angle]
data = np.loadtxt("finalNewDataNoZeros.csv", delimiter=',')

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
    # overwrite=True
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
log_dir = "./logs"
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

history = model.fit(x_train, y_train,
                    epochs=500,
                    validation_data=(x_val, y_val),
                    callbacks=[early_stopping,tensorboard_callback],
                    verbose=1)

model.save('regressionNeuralNetworkNoZeros.keras')
print("Model saved as 'regressionNeuralNetworkNoZeros.keras\n'")

# Step 6: Model Prediction
test_input = np.array([[0, 0, 0, 5, 4, 153, 2]]) / max_value

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

def compute_error(predicted, actual):
    """
    Compute absolute error for distance and angles.
    :param predicted: Predicted values [distance, start_angle, end_angle]
    :param actual: Ground truth values [distance, start_angle, end_angle]
    :return: Error array [distance_error, start_angle_error, end_angle_error]
    """
    distance_error = np.abs(predicted[0] - actual[0])
    start_angle_error = np.abs(predicted[1] - actual[1])
    end_angle_error = np.abs(predicted[2] - actual[2])
    return distance_error, start_angle_error, end_angle_error

def create_error_heatmap(predictions, ground_truth, max_distance=12, angle_increment=20, distance_increment=2):
    """
    Create a polar heatmap showing the mean error in each zone (half-circle).
    :param predictions: Model predictions [distance, start_angle, end_angle]
    :param ground_truth: Ground truth values [distance, start_angle, end_angle]
    :param max_distance: Maximum sensor detection distance
    :param angle_increment: Angular resolution for heatmap (in degrees)
    :param distance_increment: Radial resolution for heatmap (same unit as distance)
    """
    # Define angular and radial bins
    angle_bins = np.arange(0, 180 + angle_increment,
                           angle_increment)  # 0 to 180 degrees
    distance_bins = np.arange(
        2, max_distance + distance_increment, distance_increment)  # Start from 2 inches

    # Initialize the error grid and count grid
    error_grid = np.zeros((len(distance_bins) - 1, len(angle_bins) - 1))
    count_grid = np.zeros_like(error_grid)

    # Assign predictions and ground truth to the correct bins
    for pred, truth in zip(predictions, ground_truth):
        # Compute absolute errors
        distance_error = np.abs(pred[0] - truth[0])
        start_angle_error = np.abs(pred[1] - truth[1])
        end_angle_error = np.abs(pred[2] - truth[2])
        mean_error = np.mean(
            [distance_error, start_angle_error, end_angle_error])

        # Find the radial and angular bin indices
        radial_idx = np.digitize(
            truth[0], distance_bins) - 1  # Bin for distance
        angular_idx = np.digitize(
            truth[1], angle_bins) - 1    # Bin for start_angle

        # Skip out-of-bound indices (e.g., values outside the defined bins)
        if 0 <= radial_idx < len(distance_bins) - 1 and 0 <= angular_idx < len(angle_bins) - 1:
            error_grid[radial_idx, angular_idx] += mean_error
            count_grid[radial_idx, angular_idx] += 1

    # Normalize the error grid by dividing by the count grid
    with np.errstate(divide='ignore', invalid='ignore'):
        error_grid = np.divide(error_grid, count_grid, where=count_grid > 0)

    # Define the grid for plotting
    angle_grid, distance_grid = np.meshgrid(
        angle_bins[:-1], distance_bins[:-1])
    angle_grid += 10 

    # Plot the heatmap
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(10, 6))
    heatmap = ax.pcolormesh(np.radians(
        angle_grid), distance_grid, error_grid, cmap="YlOrRd", shading='auto')

    # Customize plot for a half-circle
    ax.set_theta_zero_location("E")  # 0 degrees at the top
    ax.set_theta_direction(1)  # Counterclockwise direction
    ax.set_ylim(0, max_distance)  # Start radial extent at 2 inches
    ax.set_xlim(0, np.pi)
    ax.set_yticks(distance_bins)
    ax.set_xticks(np.radians(angle_bins))  # Show ticks for 0° to 180°
    ax.set_title("Prediction Error Heatmap", va='bottom', fontsize=16)
    cbar = plt.colorbar(heatmap, ax=ax, pad=0.1)
    cbar.set_label("Mean Error", rotation=270, labelpad=20)

    plt.show()

# Create the half-circle heatmap
create_error_heatmap(predictions, ground_truth)
