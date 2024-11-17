#
# Integrated script to control the robot based on predictions from both models.
#

from irobot_edu_sdk.backend.bluetooth import Bluetooth
from irobot_edu_sdk.robots import Create3, event
import asyncio
import numpy as np
import tensorflow as tf
import os

robot_name = "Roomba"
robot = Create3(Bluetooth(robot_name))
# Load the trained models
# Ensure that the models are in the same directory or provide the correct paths
regression_model = tf.keras.models.load_model('regressionNeuralNetwork.keras')
dqn_model = tf.keras.models.load_model('dqn_robot_model.keras')

# Function to execute actions on the robot
async def execute_action(action, robot):
    if action == 0:
        print("Robot is moving forward")
        await robot.move(10)  # Move forward 10 cm
    elif action == 1:
        print("Robot is turning left")
        await robot.turn_left(15)  # Turn left 15 degrees
    elif action == 2:
        print("Robot is turning right")
        await robot.turn_right(15)  # Turn right 15 degrees
    elif action == 3:
        print("Robot is stopping")
        await robot.stop()  # Stop the robot
    else:
        print(f"Unknown action: {action}")

# Normalize sensor data as done during training
def normalize_sensor_data(sensor_data, max_value):
    return np.array(sensor_data) / max_value

# Main function to control the robot
@event(robot.when_play)
async def robot_decision_loop(robot):
    max_sensor_value=3000
    while True:
        # Get current sensor readings
        sensor_data_response = await robot.get_ir_proximity()
        sensor_data = sensor_data_response.sensors  # List of 7 sensor readings
        # Normalize sensor data
        normalized_sensor_data = normalize_sensor_data(sensor_data, max_sensor_value)
        normalized_sensor_data = normalized_sensor_data.reshape(1, -1)  # Shape (1, 7)

        # Use the regression model to predict distance and angles
        predicted_output = regression_model.predict(normalized_sensor_data)
        predicted_distance, predicted_start_angle, predicted_end_angle = predicted_output[0]

        # Prepare the state for the DQN model
        state = np.array([predicted_distance, predicted_start_angle, predicted_end_angle]).reshape(1, -1)

        # Decide action using the DQN model
        action_values = dqn_model.predict(state)
        action = np.argmax(action_values[0])

        # Execute the action
        await execute_action(action, robot)

        # Check for termination condition (e.g., distance below a threshold)
        if predicted_distance <= 5:  # Threshold distance in cm
            print("Target reached.")
            await robot.stop()
            break

        # Add a delay if necessary
        await asyncio.sleep(0.1)  # Sleep for 100 ms to avoid overwhelming the robot


# Run the main function
if __name__ == "__main__":
    robot.play()  # Start the robot