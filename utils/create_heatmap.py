import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

def custom_accuracy(y_true, y_pred):
    dist_ok = tf.less_equal(tf.abs(y_true[:, 0] - y_pred[:, 0]), 0.5)
    start_ok = tf.equal(tf.round(y_true[:, 1]), tf.round(y_pred[:, 1]))
    end_ok   = tf.equal(tf.round(y_true[:, 2]), tf.round(y_pred[:, 2]))
    all_ok = tf.logical_and(dist_ok, tf.logical_and(start_ok, end_ok))
    return tf.reduce_mean(tf.cast(all_ok, tf.float32))

def compute_error(predicted, actual):
    # Distance rule
    dist_diff = np.abs(predicted[0] - actual[0])
    distance_error = 0.0 if dist_diff <= 0.5 else dist_diff

    # Start angle rule
    start_diff = np.abs(predicted[1] - actual[1])
    start_angle_error = 0.0 if np.round(predicted[1]) == np.round(actual[1]) else start_diff

    # End angle rule
    end_diff = np.abs(predicted[2] - actual[2])
    end_angle_error = 0.0 if np.round(predicted[2]) == np.round(actual[2]) else end_diff

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
    
# Load the trained models
script_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
data_dir = os.path.join(project_root, "data")
models_dir = os.path.join(project_root, "models")
model = tf.keras.models.load_model(
    os.path.join(models_dir, "rnn.keras"),
    custom_objects={'custom_accuracy': custom_accuracy},
    compile=False 
)

data = np.loadtxt(os.path.join(data_dir, "differentSizesData/testDifferentSizesData.csv"), delimiter=',')
sensor_data = data[:, :7]
polar_coordinates = data[:, 7:]
max_value = np.max(sensor_data)

# The array of test inputs to calculate error from each bin
diagram_inputs = (sensor_data / max_value)

# Ground truth for comparison (distance, start_angle, end_angle in degrees)
ground_truth = polar_coordinates

# Normalize the the diagram inputs based on the max value from training
max_value = np.max(diagram_inputs)
diagram_inputs = diagram_inputs.astype(float) / max_value

# predict outcomes
predictions = model.predict(diagram_inputs)

# convert angles to degrees for easier comparison
predictions[:, 1:] = np.degrees(predictions[:, 1:])

# return predictions and ground truth
np.save("predictions.npy", predictions)
np.save("ground_truth.npy", ground_truth)

predictions = np.load("predictions.npy")
ground_truth = np.load("ground_truth.npy")
# Create the half-circle heatmap
create_error_heatmap(predictions, ground_truth)
