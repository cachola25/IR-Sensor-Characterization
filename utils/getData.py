import os
import sys
import signal
from irobot_edu_sdk.backend.bluetooth import Bluetooth
from irobot_edu_sdk.robots import event, Create3

prompt = "Which model is this data for?\n" \
         "(0) Regression Neural Network\n" \
         "(1) Single-Label Object Model\n> "
model = input(prompt).strip()
while model not in {"0", "1"}:
    print("Invalid input. Please enter 0 or 1.")
    model = input(prompt).strip()

script_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
data_dir = os.path.join(project_root, "data")

if model == "0":
    filename = "validation.csv"
    distance = float(input("Distance to object (in inches): "))
    start_angle = float(input("Start angle (in degrees): "))
    end_angle = float(input("End angle (in degrees): "))
else:
    filename = "a.csv"
    num_objects = int(input("Number of objects: "))

out_path = os.path.join(data_dir, filename)
out_file = open(out_path, "a", newline="")

name = "CapstoneRobot1"
robot = Create3(Bluetooth(name))
num_readings = 1
rows = 0
finished = False

@event(robot.when_play)
async def play(robot):
    global rows, finished
    while rows < num_readings:
        sensors = (await robot.get_ir_proximity()).sensors
        if sensors is None:
            print("Failed to get IR readings; retrying…")
            continue

        if model == "0":
            data_row = sensors + [distance, start_angle, end_angle]
        else:
            data_row = sensors + [num_objects]

        out_file.write(",".join(map(str, data_row)) + "\n")
        rows += 1
        print(f"Data collected: {data_row}")

    if not finished:
        await robot.play_note(440, 0.25)
        finished = True
        out_file.close()
        print("Data collection completed and file closed.")
    
robot.play()