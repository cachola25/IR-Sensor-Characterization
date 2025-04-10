import pygame
import sys
import math
import asyncio
import numpy as np
import nest_asyncio
from irobot_edu_sdk.backend.bluetooth import Bluetooth
from irobot_edu_sdk.robots import Create3
from irobot_edu_sdk.event import event
from tensorflow.keras.models import load_model
nest_asyncio.apply()

# --- Constants ---
WIDTH, HEIGHT = 1200, 700
WHITE = (255, 255, 255)
RED = (255, 0, 0)
RED_TRANSPARENT = (255, 0, 0, 80)
BLACK = (0, 0, 0)
MAX_SENSOR_VALUE = 4095
MODEL_PATH = '../combinationOfAllData.keras'
ZONE = "Zone 1"  # Example zone
OUTPUT_FILE = "roomba_data.csv"
NUM_ROWS = 100  # Example limit
TEST_MODE = False  # Set to False when near the Roomba

# --- State ---
predicted_start_deg = 90
predicted_end_deg = 90
predicted_distance = 1  # Start with a visible non-zero value
rows = 0
printed = False
out_file = open(OUTPUT_FILE, "w")
selected_zones = ZONE

# --- Pygame Setup ---
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Live Roomba Object Prediction")
clock = pygame.time.Clock()
roomba_pos = [WIDTH // 2, 600]
roomba_image = pygame.image.load("iRobot.png")
roomba_image = pygame.transform.scale(roomba_image, (350, 200))

# --- Load Model ---
model = load_model(MODEL_PATH)

# --- Roomba Setup ---
try:
    robot = Create3(Bluetooth("Roomba"))
    print("✅ Connected to Roomba via Bluetooth")
except Exception as e:
    print(f"❌ Failed to connect to Roomba: {e}")
    sys.exit(1)

# --- Roomba Event Handler ---
@event(robot.when_play)
async def play(robot):
    global rows, printed, out_file, predicted_distance, predicted_start_deg, predicted_end_deg
    print("Roomba play event triggered — starting sensor + prediction loop...")

    while True:
        try:
            sensors = await robot.get_ir_proximity()
            if sensors is None or sensors.sensors is None or len(sensors.sensors) != 7:
                print("⚠️ Skipping invalid sensor data.")
                await asyncio.sleep(0.1)
                continue
            ir_values = sensors.sensors

            print(f"IR sensors: {ir_values}")

            # Save to file
            sensor_string = f"\"{selected_zones}\",{','.join(map(str, ir_values))}\n"
            out_file.write(sensor_string)
            rows += 1

            # Make prediction
            input_data = np.array(ir_values).reshape(1, -1) / MAX_SENSOR_VALUE
            prediction = model.predict(input_data, verbose=0)[0]
            predicted_distance, start_rad, end_rad = prediction
            predicted_distance *= 550  # Scale to match grid radius
            predicted_start_deg = np.degrees(start_rad) % 180
            predicted_end_deg = np.degrees(end_rad) % 180

            print(f"🔮 Prediction -> dist: {predicted_distance:.2f}, angle: {predicted_start_deg:.2f}-{predicted_end_deg:.2f}")

            if rows >= NUM_ROWS and not printed:
                await robot.play_note(440, 0.25)
                out_file.close()
                printed = True
                print("✅ Data collection complete.")
                break

            await asyncio.sleep(0.1)

        except Exception as e:
            print(f"🔥 Error in play loop: {e}")
            await asyncio.sleep(1)

# --- Fake Event Loop for Test Mode ---
async def fake_prediction_loop():
    global rows, printed, predicted_distance, predicted_start_deg, predicted_end_deg
    print("🧪 Fake prediction loop started...")
    while True:
        ir_values = np.random.randint(200, 3000, size=7).tolist()
        input_data = np.array(ir_values).reshape(1, -1) / MAX_SENSOR_VALUE
        prediction = model.predict(input_data, verbose=0)[0]
        predicted_distance, start_rad, end_rad = prediction
        predicted_distance *= 550  # Scale to match grid radius
        predicted_start_deg = np.degrees(start_rad) % 180
        predicted_end_deg = np.degrees(end_rad) % 180

        print(f"🧪 [TEST] dist={predicted_distance:.2f}, angle={predicted_start_deg:.2f}-{predicted_end_deg:.2f}")
        await asyncio.sleep(0.5)

# --- Pygame Loop ---
async def run_game():
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill(WHITE)
        screen.blit(roomba_image, (roomba_pos[0] - roomba_image.get_width() // 2, roomba_pos[1] - roomba_image.get_height() // 2))
        draw_arc_with_grid(screen, roomba_pos, 550, 0, 180, 20)
        draw_prediction_cone(screen, roomba_pos, 550, predicted_start_deg, predicted_end_deg, predicted_distance)

        print(f"[DRAW] dist={predicted_distance:.2f}, start={predicted_start_deg:.2f}, end={predicted_end_deg:.2f}")

        pygame.display.flip()
        clock.tick(30)
        await asyncio.sleep(0)

    pygame.quit()
    sys.exit()

# --- Draw Arc Grid ---
def draw_arc_with_grid(surface, center, radius, start_angle, end_angle, sector_angle):
    arc_surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
    arc_surface.fill((0, 0, 0, 0))
    rect = pygame.Rect(0, 0, radius * 2, radius * 2)
    pygame.draw.arc(arc_surface, BLACK, rect, math.radians(start_angle), math.radians(end_angle), 2)
    for angle in range(start_angle, end_angle + 1, sector_angle):
        end_x = radius + radius * math.cos(math.radians(angle))
        end_y = radius - radius * math.sin(math.radians(angle))
        pygame.draw.line(arc_surface, BLACK, (radius, radius), (end_x, end_y), 1)
        angle_rad = math.radians(angle)
        label_x = end_x + 10 * math.cos(angle_rad)
        label_y = end_y - 10 * math.sin(angle_rad)
        label = pygame.font.SysFont(None, 20).render(f"{angle}°", True, BLACK)
        arc_surface.blit(label, (label_x - 10, label_y - 10))

    offset = 0
    for i in range(1, 7):
        r = int(offset + (radius - offset) * i / 6)
        pygame.draw.circle(arc_surface, BLACK, (radius, radius), r, 1)
        if i >= 2:
            inches = (i - 1) * 2
            label = pygame.font.SysFont(None, 20).render(f"{inches}in", True, BLACK)
            arc_surface.blit(label, (radius - 20, radius - r + 5))

    mask = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
    mask.fill((0, 0, 0, 0))
    pygame.draw.rect(mask, (255, 255, 255, 255), (0, 0, radius * 2, radius + 5))
    arc_surface.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
    surface.blit(arc_surface, (center[0] - radius, center[1] - radius))

# --- Draw Prediction Cone ---
def draw_prediction_cone(surface, center, radius, start_angle, end_angle, distance):
    arc_surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
    arc_surface.fill((0, 0, 0, 0))
    distance = max(5, min(distance, radius))  # Prevent drawing a dot only
    start_x = radius + distance * math.cos(math.radians(start_angle))
    start_y = radius - distance * math.sin(math.radians(start_angle))
    end_x = radius + distance * math.cos(math.radians(end_angle))
    end_y = radius - distance * math.sin(math.radians(end_angle))
    pygame.draw.polygon(arc_surface, RED_TRANSPARENT, [(radius, radius), (start_x, start_y), (end_x, end_y)])
    pygame.draw.line(arc_surface, RED, (radius, radius), (start_x, start_y), 3)
    pygame.draw.line(arc_surface, RED, (radius, radius), (end_x, end_y), 3)
    mask = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
    mask.fill((0, 0, 0, 0))
    pygame.draw.rect(mask, (255, 255, 255, 255), (0, 0, radius * 2, radius + 5))
    arc_surface.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
    surface.blit(arc_surface, (center[0] - radius, center[1] - radius))

# --- Main Runner ---
async def main():
    try:
        if not TEST_MODE:
            print("🔌 Attempting to connect to Roomba...")
            robot.play()
            print("✅ Roomba play triggered! Press the Play button on the robot if you haven’t already.")

            tasks = [run_game(), asyncio.Event().wait()]
        else:
            print("🧪 TEST MODE: Running with fake sensor data!")
            tasks = [run_game(), fake_prediction_loop()]

        await asyncio.gather(*[asyncio.create_task(t) for t in tasks])

    except Exception as e:
        print(f"🔥 Could not connect to Roomba: {e}")

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
