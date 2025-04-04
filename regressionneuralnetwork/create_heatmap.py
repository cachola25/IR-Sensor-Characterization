import numpy as np
import matplotlib.pyplot as plt

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

predictions = np.load("predictions.npy")
ground_truth = np.load("ground_truth.npy")
# Create the half-circle heatmap
create_error_heatmap(predictions, ground_truth)
