import os
import numpy as np
import pandas as pd
import tensorflow as tf
import keras_tuner as kt
from tensorflow import keras
import matplotlib.pyplot as plt
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
                metrics=['mse', custom_accuracy])
    return model

# Define a custom accuracy metric
"""
dist_ok = is predicted distance within 1/2 an inch of the actual distance?
start_ok = did the model predict the whole number of the start angle correctly?
end_ok = did the model predict the whole number of the end angle correctly?
"""
def custom_accuracy(y_true, y_pred):
    dist_ok = tf.less_equal(tf.abs(y_true[:, 0] - y_pred[:, 0]), 0.5)
    start_ok = tf.equal(tf.round(y_true[:, 1]), tf.round(y_pred[:, 1]))
    end_ok   = tf.equal(tf.round(y_true[:, 2]), tf.round(y_pred[:, 2]))
    all_ok = tf.logical_and(dist_ok, tf.logical_and(start_ok, end_ok))
    return tf.reduce_mean(tf.cast(all_ok, tf.float32))

# Load the data and model folder
script_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
data_dir = os.path.join(project_root, "data")
models_dir = os.path.join(project_root, "models")

# Step 1: Load Data from CSV
file_path_original = os.path.join(data_dir,"pca_combinationOfAllData.csv")
data_original = pd.read_csv(file_path_original, header=None)

# Shuffle the rows and convert directly to NumPy
data = data_original.sample(frac=1, random_state=42).reset_index(drop=True).to_numpy()
# data = data_original.to_numpy()

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
    directory='tuning_dir',
    project_name='ir_sensor_tuning',
    overwrite=False
)

# Early stopping to avoid overfitting
early_stopping = EarlyStopping(
    monitor='val_loss', patience=50, restore_best_weights=True)

# Split the data into training and validation sets
split_index = int(0.8 * len(sensor_data))
x_train, x_val = sensor_data[:split_index], sensor_data[split_index:]
y_train, y_val = polar_coordinates[:split_index], polar_coordinates[split_index:]

print(f"Beginning tuning process...")

# Step 4: Perform hyperparameter search
tuner.search(x_train, y_train,
             epochs=100,
             validation_data=(x_val, y_val),
             callbacks=[early_stopping],
             verbose=1,
             )

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
                    verbose=1)

# Save the model to the model's folder
model_name = "rnn.keras"
filename = os.path.join(models_dir, model_name)
model.save(filename)
print(f"Model saved as {model_name}\n")

# Step 6: Model Prediction
# Evaluate the model on the validation set
val_loss, val_mse, val_accuracy = model.evaluate(x_val, y_val, verbose=1)
print(f"Validation Loss (MSE): {val_loss:.4f}")
print(f"Validation MSE: {val_mse:.4f}")
print(f"Custom Accuracy: {val_accuracy:.4f}")
