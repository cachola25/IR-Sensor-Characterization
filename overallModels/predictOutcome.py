import asyncio
import time
import numpy as np
import tensorflow as tf
from overallModel import overallModel
# Measure total script execution time
script_start_time = time.time()

# load single-label model to predict number of objects

# Load and preprocess the data
test_input = np.array([[1,28,97,75,16,101,11]], dtype=float)
final_model = overallModel()
prediction = final_model.predict(test_input)
print(prediction)
print(f"Total script execution time: {time.time() - script_start_time:.4f} seconds.")