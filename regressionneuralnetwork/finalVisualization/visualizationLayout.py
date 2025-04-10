import pygame
import asyncio
import nest_asyncio
from irobot_edu_sdk.backend.bluetooth import Bluetooth
from irobot_edu_sdk.robots import Create3

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

# --- Pygame Setup ---
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Roomba IR Viewer")
clock = pygame.time.Clock()

# --- Draw IR Values ---
def draw_ir_data(surface, data):
    font = pygame.font.SysFont(None, 32)
    y = 50
    for i, val in enumerate(data):
        label = font.render(f"IR {i+1}: {val}", True, (0, 0, 0))
        surface.blit(label, (50, y))
        y += 40

# --- Pygame Loop ---
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

# --- Roomba Collector ---
# --- Roomba Collector ---
async def collect_ir_data(robot):
    global rows, ir_values
    print("📡 Starting IR collection (no play button required)...")
    while rows < NUM_ROWS and window_open:
        try:
            sensors = await robot.get_ir_proximity()
            print(f"🧪 Raw sensor result: {sensors}")  # ← DEBUG LINE

            if sensors and sensors.sensors and len(sensors.sensors) == 7:
                ir_values[:] = sensors.sensors
                rows += 1
                print(f"📦 Row {rows}: {ir_values}")
            else:
                print("⚠️ Skipped invalid sensor data")
        except Exception as e:
            print(f"🔥 IR fetch error: {e}")
        await asyncio.sleep(0.1)

    print("✅ Collected 100 rows. You can still view them in the window.")


# --- Main ---
async def main():
    print("🔌 Connecting to Roomba...")
    robot = Create3(Bluetooth(ROBOT_NAME))
    print("✅ Connected to Roomba")

    pygame_task = asyncio.create_task(run_pygame())
    roomba_task = asyncio.create_task(collect_ir_data(robot))

    await pygame_task

    if not roomba_task.done():
        roomba_task.cancel()
        try:
            await roomba_task
        except asyncio.CancelledError:
            print("🛑 Roomba task cancelled")

    print("✅ Clean exit")

if __name__ == "__main__":
    asyncio.run(main())
