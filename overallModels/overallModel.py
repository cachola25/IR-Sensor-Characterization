import time
import numpy as np
import tensorflow as tf
import os
class overallModel:
    def __init__(self):
        script_dir = os.path.dirname(os.path.realpath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, ".."))
        data_dir = os.path.join(project_root, "data")
        models_dir = os.path.join(project_root, "models")
        self.single_label_model = tf.keras.models.load_model(
            os.path.join(models_dir, "single_label.keras"),
            compile=False 
        )
        self.regression_model = tf.keras.models.load_model(
            os.path.join(models_dir, "rnn.keras"),
            compile=False 
        )
        self.data = np.loadtxt(os.path.join(data_dir, "multi_object_data.csv"), delimiter=',')
        self.sensor_data = self.data[:, :7]
        self.max_value = np.max(self.sensor_data)
        
        
    # Finds indices of the two highest peaks that are at least min_separation apart
    def find_two_peaks(self,sensor_values, min_separation=1):
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
    def create_masked_input(self,sensor_values, peak_idx, window_size=1, mask_value=3):

        masked = np.full_like(sensor_values, fill_value=mask_value)
        start = max(0, peak_idx - window_size)
        end = min(len(sensor_values), peak_idx + window_size + 1)
        masked[start:end] = sensor_values[start:end]
        return masked
    
    def predict(self,ir_data):
        copied_input = ir_data.copy()
        predicted_probabilities = self.single_label_model.predict(ir_data,verbose=0)[0]
        predicted_class = np.argmax(predicted_probabilities)
        # print(f"Predicted Number of Objects: {predicted_class}")
        # print(f"Predicted Probabilites: {predicted_probabilities}")
        ret = []

        if predicted_class != 0:
            if predicted_class == 1:

                predicted_output = self.regression_model.predict(ir_data[0],verbose=0)
                predicted_distance, predicted_start_angle, predicted_end_angle = predicted_output[0]
                predicted_start_angle_deg = np.degrees(predicted_start_angle)
                predicted_end_angle_deg = np.degrees(predicted_end_angle)

                if predicted_start_angle_deg > 180:
                    predicted_start_angle_deg -= 180
                if predicted_end_angle_deg > 180:
                    predicted_end_angle_deg -= 180

                # Print predictions
                # print(f"Predicted Distance: {predicted_distance}")
                # print(f"Start Angle: {predicted_start_angle_deg} degrees")
                # print(f"End Angle: {predicted_end_angle_deg} degrees")
                ret.append((predicted_distance,predicted_start_angle_deg,predicted_end_angle_deg))

            elif predicted_class == 2:
                peak1, peak2 = self.find_two_peaks(copied_input[0])
            
                # first peak
                input1 = self.create_masked_input(copied_input[0], peak1)
                # print(f"[Object 1] Masked IR values before normalization: {input1}")
                input1_normalized = input1 / self.max_value
                predicted_output = self.regression_model.predict(input1_normalized[np.newaxis, :],verbose=0)

                predicted_distance, predicted_start_angle, predicted_end_angle = predicted_output[0]
                predicted_start_angle_deg = np.degrees(predicted_start_angle)
                predicted_end_angle_deg = np.degrees(predicted_end_angle)

                if predicted_start_angle_deg > 180:
                    predicted_start_angle_deg -= 180
                if predicted_end_angle_deg > 180:
                    predicted_end_angle_deg -= 180

                # Print predictions
                # print(f"[Object 1] Predicted Distance: {predicted_distance:.2f} inches; Start Angle: {predicted_start_angle_deg:.2f} degrees; End Angle: {predicted_end_angle_deg:.2f} degrees")
                ret.append((predicted_distance,predicted_start_angle_deg,predicted_end_angle_deg))

                # second peak
                input2 = self.create_masked_input(copied_input[0], peak2)
                # print(f"[Object 1] Masked IR values before normalization: {input2}")
                input2_normalized = input2 / self.max_value
                predicted_output = self.regression_model.predict(input2_normalized[np.newaxis, :],verbose=0)

                predicted_distance, predicted_start_angle, predicted_end_angle = predicted_output[0]
                predicted_start_angle_deg = np.degrees(predicted_start_angle)
                predicted_end_angle_deg = np.degrees(predicted_end_angle)

                if predicted_start_angle_deg > 180:
                    predicted_start_angle_deg -= 180
                if predicted_end_angle_deg > 180:
                    predicted_end_angle_deg -= 180

                # Print predictions
                # print(f"[Object 2] Predicted Distance: {predicted_distance:.2f} inches; Start Angle: {predicted_start_angle_deg:.2f} degrees; End Angle: {predicted_end_angle_deg:.2f} degrees")
                ret.append((predicted_distance,predicted_start_angle_deg,predicted_end_angle_deg))
        return ret
