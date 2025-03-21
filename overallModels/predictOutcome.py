import asyncio
import time
import numpy as np
import tensorflow as tf

# Finds indices of the two highest peaks that are at least min_separation apart
def find_two_peaks(sensor_values, min_separation=1):
    peaks = sensor_values.copy()
    first_peak_idx = np.argmax(peaks)
    first_peak_val = peaks[first_peak_idx]
    
    # Mask out a region around the first peak to find the second one
    start = max(0, first_peak_idx - min_separation)
    end = min(len(peaks), first_peak_idx + min_separation + 1)
    peaks[start:end] = -np.inf  # Set nearby region to -inf so it's ignored
    
    second_peak_idx = np.argmax(peaks)
    second_peak_val = peaks[second_peak_idx]

    return first_peak_idx, second_peak_idx


 # Masks sensor values to isolate peaks and sets other values mask_value
def create_masked_input(sensor_values, peak_idx, window_size=1, mask_value=3):

    masked = np.full_like(sensor_values, fill_value=mask_value)
    start = max(0, peak_idx - window_size)
    end = min(len(sensor_values), peak_idx + window_size + 1)
    masked[start:end] = sensor_values[start:end]
    return masked



# Measure total script execution time
script_start_time = time.time()

# load single-label model to predict number of objects
model = tf.keras.models.load_model('outcomePredictionModel3.keras')

# Load and preprocess the data
test_input = np.array([[1,28,97,75,16,101,11]], dtype=float)
copied_input = test_input.copy()
data = np.loadtxt("test3.csv", delimiter=',')
sensor_data = data[:, :7]
max_value = np.max(sensor_data)
test_input /= max_value 

predicted_probabilities = model.predict(test_input)[0]
predicted_class = np.argmax(predicted_probabilities)

# Print prediction result
print(f"Predicted Number of Objects: {predicted_class}")
print(f"Predicted Probabilites: {predicted_probabilities}")

if predicted_class != 0:
    # load regression-based model to predict distance and angle 
    model = tf.keras.models.load_model('differentSizes2.keras')

    if predicted_class == 1:

        predicted_output = model.predict(test_input[0])
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

    elif predicted_class == 2:
        peak1, peak2 = find_two_peaks(copied_input[0])
    
        # first peak
        input1 = create_masked_input(copied_input[0], peak1)
        print(f"[Object 1] Masked IR values before normalization: {input1}")
        input1_normalized = input1 / max_value
        predicted_output = model.predict(input1_normalized[np.newaxis, :])

        predicted_distance, predicted_start_angle, predicted_end_angle = predicted_output[0]
        predicted_start_angle_deg = np.degrees(predicted_start_angle)
        predicted_end_angle_deg = np.degrees(predicted_end_angle)

        if predicted_start_angle_deg > 180:
            predicted_start_angle_deg -= 180
        if predicted_end_angle_deg > 180:
            predicted_end_angle_deg -= 180

        # Print predictions
        print(f"[Object 1] Predicted Distance: {predicted_distance:.2f} inches; Start Angle: {predicted_start_angle_deg:.2f} degrees; End Angle: {predicted_end_angle_deg:.2f} degrees")

        # second peak

        input2 = create_masked_input(copied_input[0], peak2)
        print(f"[Object 1] Masked IR values before normalization: {input2}")
        input2_normalized = input2 / max_value
        predicted_output = model.predict(input2_normalized[np.newaxis, :])

        predicted_distance, predicted_start_angle, predicted_end_angle = predicted_output[0]
        predicted_start_angle_deg = np.degrees(predicted_start_angle)
        predicted_end_angle_deg = np.degrees(predicted_end_angle)

        if predicted_start_angle_deg > 180:
            predicted_start_angle_deg -= 180
        if predicted_end_angle_deg > 180:
            predicted_end_angle_deg -= 180

        # Print predictions
        print(f"[Object 2] Predicted Distance: {predicted_distance:.2f} inches; Start Angle: {predicted_start_angle_deg:.2f} degrees; End Angle: {predicted_end_angle_deg:.2f} degrees")

    
print(f"Total script execution time: {time.time() - script_start_time:.4f} seconds.")