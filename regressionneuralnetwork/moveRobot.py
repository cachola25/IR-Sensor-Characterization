#
# Integrated script to control the robot based on predictions from both models.
#
import asyncio
import numpy as np
import tensorflow as tf

from irobot_edu_sdk.robots import Create3, event
from irobot_edu_sdk.backend.bluetooth import Bluetooth

robot_name = "Roomba"
robot = Create3(Bluetooth(robot_name))
# Load the trained models
regression_model = tf.keras.models.load_model('regressionNeuralNetwork.keras')
dqn_model = tf.keras.models.load_model('dqn_robot_model.keras')

# Function to execute actions on the robot
async def execute_action(action, robot):
    if action == 0:
        print("Robot is moving forward")
        await robot.move(5)  # Move forward 10 cm
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
    max_sensor_value=2087
    while True:
        # Get current sensor readings
        sensor_data_response = await robot.get_ir_proximity()
        sensor_data = sensor_data_response.sensors
        
        # Normalize sensor data
        normalized_sensor_data = normalize_sensor_data(sensor_data, max_sensor_value)
        normalized_sensor_data = normalized_sensor_data.reshape(1, -1)  # Shape (1, 7)

        # Use the regression model to predict distance and angles
        predicted_output = regression_model.predict(normalized_sensor_data)
        predicted_distance, predicted_start_angle, predicted_end_angle = predicted_output[0]
        
        # Convert distance output to cm and angles to degrees
        predicted_distance *= 2.54
        predicted_start_angle = np.degrees(predicted_start_angle)
        predicted_end_angle = np.degrees(predicted_end_angle)
        
        print(f"DISTANCE: {predicted_distance}")
        print(f"START_ANGLE: {predicted_start_angle}")
        print(f"END_ANGLE: {predicted_end_angle}")
        
        # Prepare the state for the DQN model
        state = np.array([predicted_distance, predicted_start_angle, predicted_end_angle]).reshape(1, -1)

        # Decide action using the DQN model
        action_values = dqn_model.predict(state,verbose=0)
        action = np.argmax(action_values[0])

        # Execute the action
        await execute_action(action, robot)

        if predicted_distance <= 5:  # Threshold distance in cm (~2 in)
            print("Target reached.")

        # Add a delay if necessary
        await asyncio.sleep(0.1)  # Sleep for 100 ms to avoid overwhelming the robot

robot.play()  # Start the robot