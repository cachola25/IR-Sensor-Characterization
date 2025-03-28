import pygame
import sys
import random
import math

# Initialize Pygame
pygame.init()

# Screen settings
WIDTH, HEIGHT = 1200, 700
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Roomba Object Prediction")

# Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)

# Clock for controlling frame rate
clock = pygame.time.Clock()

# Roomba position
roomba_pos = [WIDTH // 2, 600]

# Load Roomba image
roomba_image = pygame.image.load("iRobot.png")  # Replace with your image file path
roomba_image = pygame.transform.scale(roomba_image, (315, 175))  # Scale the image

# Simulated predicted position (updates randomly)
def get_predicted_position():
    return [random.randint(50, WIDTH - 50), random.randint(50, HEIGHT - 50)]

predicted_pos = get_predicted_position()

def draw_arc_with_sectors(surface, center, radius, start_angle, end_angle, sector_angle, distance_increment):
    # Create a new surface for the arc
    arc_surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
    arc_surface.fill((0, 0, 0, 0))  # Fill with transparent color

    # Draw the arc on the new surface
    rect = pygame.Rect(0, 0, radius * 2, radius * 2)
    pygame.draw.arc(arc_surface, BLACK, rect, math.radians(start_angle), math.radians(end_angle), 2)
    
    # Draw the sectors
    for angle in range(start_angle, end_angle + 1, sector_angle):
        end_x = radius + radius * math.cos(math.radians(angle))
        end_y = radius - radius * math.sin(math.radians(angle))
        pygame.draw.line(arc_surface, BLACK, (radius, radius), (end_x, end_y), 1)
    
    # Draw the distance increments
    for i in range(1, 6):  # 5 increments
        increment_radius = radius * (i / 5)
        pygame.draw.circle(arc_surface, BLACK, (radius, radius), int(increment_radius), 1)
    
    # Create a mask to clip the bottom half of the circle everything beyond 180 to 0 degrees
    mask = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
    mask.fill((0, 0, 0, 0))  # Fill with transparent color
    pygame.draw.rect(mask, (255, 255, 255, 255), (0, 0, radius * 2, radius + 5))
    
    # Apply the mask to the arc surface
    arc_surface.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
    
    # Blit the arc surface onto the main surface
    surface.blit(arc_surface, (center[0] - radius, center[1] - radius))

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    # Simulate prediction update (replace with actual sensor data)
    predicted_pos = get_predicted_position()
    
    # Clear the screen
    screen.fill(WHITE)
    
    # Draw Roomba (using the image)
    screen.blit(roomba_image, (roomba_pos[0] - roomba_image.get_width() // 2, roomba_pos[1] - roomba_image.get_height() // 2))
    
    # Draw Predicted Object
    pygame.draw.circle(screen, RED, predicted_pos, 15)
    
    # Draw Prediction Line
    pygame.draw.line(screen, GREEN, roomba_pos, predicted_pos, 3)
    
    # Draw the arc with sectors and distance increments
    draw_arc_with_sectors(screen, roomba_pos, 550, 0, 180, 20, 2)  
    
    # Update the display
    pygame.display.flip()
    
    # Control frame rate
    clock.tick(30)

# Quit Pygame
pygame.quit()
sys.exit()