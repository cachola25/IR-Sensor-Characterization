import pygame
import asyncio
import nest_asyncio
from irobot_edu_sdk.backend.bluetooth import Bluetooth
from irobot_edu_sdk.robots import Create3
from irobot_edu_sdk.event import event

nest_asyncio.apply()

# --- Constants ---
WIDTH, HEIGHT = 800, 600
WHITE = (255, 255, 255)
ROBOT_NAME = "CapstoneRobot1"
NUM_ROWS = 100

# --- State ---
ir_values = [0] * 7
rows = 0
window_open = True
printed = False

# --- Pygame Setup ---
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Roomba IR Viewer")
clock = pygame.time.Clock()

def draw_ir_data(surface, data):
    font = pygame.font.SysFont(None, 32)
    y = 50
    for i, val in enumerate(data):
        label = font.render(f"IR {i+1}: {val}", True, (0, 0, 0))
        surface.blit(label, (50, y))
        y += 40

async def run_pygame():
    global window_open
    try:
        while window_open:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    window_open = False
            screen.fill(WHITE)
            draw_ir_data(screen, ir_values)
            pygame.display.flip()
            clock.tick(30)
            await asyncio.sleep(0)
    except asyncio.CancelledError:
        print("🛑 Pygame loop cancelled")
    finally:
        pygame.quit()

async def collect_data(robot):
    global rows, ir_values
    print("📡 IR collection running...")
    while rows < NUM_ROWS and window_open:
        sensors = await robot.get_ir_proximity()
        if sensors and sensors.sensors and len(sensors.sensors) == 7:
            ir_values[:] = sensors.sensors
            rows += 1
            print(f"📦 Row {rows}: {ir_values}")
        else:
            print("⚠️ Skipped invalid sensor data")
        await asyncio.sleep(0.1)

@event(robot := Create3(Bluetooth(ROBOT_NAME)).when_play)
async def handle_play(robot):
    print("▶️ Play button pressed. Starting collection & visualization...")
    try:
        # Run both tasks together
        await asyncio.gather(
            run_pygame(),
            collect_data(robot)
        )
    except asyncio.CancelledError:
        print("🛑 Tasks cancelled")
    finally:
        await robot.play_note(440, 0.25)
        print("✅ Done — shutting down")

# --- Start the robot (and wait for Play) ---
print("🔌 Connecting to Roomba...")
print("✅ Connected — waiting for Play button...")
Create3(Bluetooth(ROBOT_NAME)).play()
