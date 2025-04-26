import numpy as np
import tensorflow as tf
import joblib
import os
import pandas as pd
from overallModel import overallModel

script_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../"))
data_dir = os.path.join(project_root, "data")
models_dir = os.path.join(project_root, "models")
pca = joblib.load(os.path.join(models_dir, "rnn_pca_model.joblib"))
raw = pd.read_csv(os.path.join(data_dir, "pca_combinationOfAllData.csv"), header=None).to_numpy()
raw_matrix = pd.read_csv(os.path.join(data_dir, "pca_combinationOfAllData.csv"), header=None).to_numpy()
pca_space = raw_matrix[:, :7]
PCA_MAX = float(np.max(pca_space))

# Load and preprocess the data

# raw_in  = np.array(ir_values).reshape(1, -1)
# pca_in  = pca.transform(raw_in)
# norm_pca = pca_in / PCA_MAX
# dist_in, start_rad, end_rad = model.predict(norm_pca, verbose=0)[0]

raw_in = np.array([5,5,19,12,16,3,12]).reshape(1,-1)
pca_in = pca.transform(raw_in)
norm_pca = pca_in / PCA_MAX
# test_input = np.array([norm_pca], dtype=float)

final_model = overallModel()
prediction = final_model.predict(norm_pca)

print(f"Model predicted: {len(prediction)} objects")
for i, pred in enumerate(prediction):
    predicted_distance, predicted_start_angle_deg, predicted_end_angle_deg = pred
    print(f"[Object {i+1}] Predicted Distance: {predicted_distance:.2f} inches; Start Angle: {predicted_start_angle_deg:.2f} degrees; End Angle: {predicted_end_angle_deg:.2f} degrees")