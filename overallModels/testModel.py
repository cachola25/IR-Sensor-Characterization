import numpy as np
import tensorflow as tf
from overallModel import overallModel

# Load and preprocess the data
test_input = np.array([[1,28,97,75,16,101,11]], dtype=float)
final_model = overallModel()
prediction = final_model.predict(test_input)

print(f"Model predicted: {len(prediction)} objects")
for i, pred in enumerate(prediction):
    predicted_distance, predicted_start_angle_deg, predicted_end_angle_deg = pred
    print(f"[Object {i+1}] Predicted Distance: {predicted_distance:.2f} inches; Start Angle: {predicted_start_angle_deg:.2f} degrees; End Angle: {predicted_end_angle_deg:.2f} degrees")