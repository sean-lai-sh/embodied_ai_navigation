import json
import pygame
import cv2
import numpy as np

# Initialize Pygame
pygame.init()

# Screen setup
width, height = 400, 400
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Maze Exploration Map")
clock = pygame.time.Clock()

# Load exploration data
with open("./data/data_info.json", "r") as f:
    data = json.load(f)

# Parameters
direction = 0  # Angle in degrees (0=UP)
x, y = width // 2, height // 2
step_size = 5
scale = 1

# Surfaces
minimap = pygame.Surface((width, height))
minimap.fill((107, 107, 107))

wall_minimap = pygame.Surface((width, height), pygame.SRCALPHA)
wall_minimap.fill((0, 0, 0, 0))

# Functions clearly defined
def map_walls(image, threshold=100):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    points = np.vstack(contours).squeeze()
    return points

def transform_points(points, pos, angle, scale=1):
    angle_rad = np.deg2rad(angle)
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad),  np.cos(angle_rad)]
    ])
    points = np.dot(points, rotation_matrix) * scale
    points += pos
    return points.astype(int)

def draw_walls(surface, wall_points):
    for point in wall_points:
        pygame.draw.circle(surface, (0, 0, 255), point, 2)

# Path tracking
path = [(x, y)]
data_index = 0  # To track current data index
running = True

# Main Loop
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    if data_index < len(data):
        entry = data[data_index]
        actions = entry['action']
        image_path = entry['image']

        fpv = cv2.imread(f"./data/images/{image_path}")
        if fpv is not None:
            wall_points = map_walls(fpv)
            transformed_walls = transform_points(wall_points, np.array([x, y]), direction, scale)
            draw_walls(wall_minimap, transformed_walls)

        for action in actions:
            if action == "FORWARD":
                rad_angle = np.deg2rad(direction)
                x += step_size * np.sin(rad_angle)
                y -= step_size * np.cos(rad_angle)
                path.append((x, y))
            elif action == "RIGHT":
                direction = (direction + 90) % 360
            elif action == "LEFT":
                direction = (direction - 90) % 360

        data_index += 1

    screen.fill((255, 255, 255))
    screen.blit(minimap, (0, 0))
    screen.blit(wall_minimap, (0, 0))

    # Draw path
    for i in range(1, len(path)):
        pygame.draw.line(screen, (0, 0, 255), path[i - 1], path[i], 2)

    # Draw start and end points
    pygame.draw.circle(screen, (0, 255, 0), path[0], 8)
    pygame.draw.circle(screen, (255, 0, 0), path[-1], 8)

    pygame.display.flip()
    clock.tick(30)

pygame.quit()
