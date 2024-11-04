#
# Licensed under 3-Clause BSD license available in the License file. Copyright (c) 2021-2022 iRobot Corporation. All rights reserved.
#
from irobot_edu_sdk.backend.bluetooth import Bluetooth
from irobot_edu_sdk.robots import event, Create3
import asyncio
import os
import csv

# Prompt for the output file name and zones
filename = "ir_sensor_data"

zones = input("Enter the grid squares the object is in (e.g., a1,a2,a3,b3): ").split(",")
filename = f"{filename}.csv"

# Check if the file already exists and open for writing
file_exists = os.path.isfile(filename)
out_file = open(filename, "a", newline='')

name = "CapstoneRobot1"
robot = Create3(Bluetooth(name))
num_readings = 100  # Number of readings to collect
rows = 0
printed = False

@event(robot.when_play)
async def play(robot):
    global rows, printed
    while rows < num_readings:

        # Get IR sensor readings
        sensors = (await robot.get_ir_proximity()).sensors
        if sensors is None:
            print("Failed to get IR sensor readings.")
            continue 

        # Create an array initialized to 0
        occupancy_vector = [0] * 153

        # Update array based on the input zones
        for zone in zones: 

            # Convert the zone index and mark 1 with object inside and 0 for no object inside 
            column_index = (ord(zone[0]) - ord('a')) + (int(zone[1:]) - 1) * 17
            if 0 <= column_index < 153:  # Check if the index is within range
                occupancy_vector[column_index] = 1

        # Combine sensor readings with occupancy data and write to csv
        sensor_data = sensors + occupancy_vector 
        out_file.write(",".join(map(str, sensor_data)) + "\n")
        rows += 1

        print(f"Data collected: {sensor_data}")

        # await asyncio.sleep(0.1)

    # Close the file
    if not printed:
        await robot.play_note(440, 0.25)
        printed = True
        out_file.close()
        print("Data collection completed. File closed.")

# Start the robot
robot.play()
