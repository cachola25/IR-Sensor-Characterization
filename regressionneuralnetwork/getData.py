# Licensed under 3-Clause BSD license available in the License file. Copyright (c) 2021-2022 iRobot Corporation. All rights reserved.

from irobot_edu_sdk.backend.bluetooth import Bluetooth
from irobot_edu_sdk.robots import event, Create3
import asyncio
import os

filename = "c.csv"

# Check if the file already exists and open for writing
file_exists = os.path.isfile(filename)
out_file = open(filename, "a", newline='')

# Prompt for polar coordinates
distance = float(input("Enter the distance to the object (in inches): "))
start_angle = float(input("Enter the start angle of the object (in degrees): "))
end_angle = float(input("Enter the end angle of the object (in degrees): "))

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

        # Combine sensor readings with user-provided polar coordinates
        data_row = sensors + [distance, start_angle, end_angle]
        out_file.write(",".join(map(str, data_row)) + "\n")
        rows += 1

        print(f"Data collected: {data_row}")


    # Close the file
    if not printed:
        await robot.play_note(440, 0.25)
        printed = True
        out_file.close()
        print("Data collection completed. File closed.")

# Start the robot
robot.play()
