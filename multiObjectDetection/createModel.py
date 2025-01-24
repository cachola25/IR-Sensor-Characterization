import numpy as np
import tensorflow as tf
import keras_tuner as kt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

# Define a model-building function for Keras Tuner
def build_model(hp):
    model = keras.Sequential()
    model.add(keras.Input(shape=(7,)))  # Input layer for 7 IR sensor readings
    
    # Tune the number of hidden layers
    for i in range(hp.Int('num_layers', 1, 3)):
        # Tune the number of units in each layer
        units = hp.Int(f'units_{i}', min_value=16, max_value=512, step=16)
        model.add(layers.Dense(units=units, activation='relu'))
    
    model.add(layers.Dense(1))  # Output layer: single neuron for `num_objects`
    
    # Tune the learning rate for the optimizer
    learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='mean_squared_error',
                  metrics=['mae'])
    return model

# Step 1: Load Data from CSV 
# [sensor1, sensor2, ..., sensor7, num_objects]
data = np.loadtxt("test.csv", delimiter=',')

# Step 2: Separate the Data
sensor_data = data[:, :7]
num_objects = data[:, 7]

# Normalize the sensor data by dividing by the largest recorded sensor value
max_value = np.max(sensor_data)
sensor_data /= max_value

# Step 3: Define Model and Tuner
tuner = kt.RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=10,
    executions_per_trial=1,
    directory='my_dir',
    project_name='object_detection_tuning'
)

# Early stopping to avoid overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Split the data into training and validation sets
split_index = int(0.8 * len(sensor_data))
x_train, x_val = sensor_data[:split_index], sensor_data[split_index:]
y_train, y_val = num_objects[:split_index], num_objects[split_index:]

# Step 4: Perform hyperparameter search
print("Beginning tuning process...")
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
                    verbose=1)

model.save('objectDetectionModel.keras')
print("Model saved as 'objectDetectionModel.keras'")

# Step 6: Model Prediction
test_input = np.array([[0,4,1,10,3,4,0]]) / max_value

predicted_objects = model.predict(test_input)[0][0]
print(f"Predicted Number of Objects: {predicted_objects}")
