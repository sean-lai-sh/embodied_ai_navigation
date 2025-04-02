import json
import pygame
import cv2
import numpy as np

# Initialize Pygame and setup the window
pygame.init()
width, height = 800, 800
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Maze Exploration Map")
clock = pygame.time.Clock()

# Load exploration steps from a JSON file
with open("data_info.json", "r") as f:
    data = json.load(f)

# Movement parameters
x, y = 0, 0  # Start at virtual origin
direction = 0  # Facing up
step_size = 5
zoom = 1.0  # Initial zoom level
zoom_step = 0.1
pan_x, pan_y = 0, 0

# Virtual canvas to hold the full explored area
canvas_size = 5000
virtual_canvas = pygame.Surface((canvas_size, canvas_size))
virtual_canvas.fill((107, 107, 107))

# Persistent path layer (only draw to this when path changes)
path_surface = pygame.Surface((canvas_size, canvas_size), pygame.SRCALPHA)

# Track positions
path = [(x, y)]
data_index = 0
running = True

while running:
    dirty = False  # Only update path if movement occurred

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_EQUALS or event.key == pygame.K_KP_PLUS:
                zoom = min(2.0, zoom + zoom_step)
            elif event.key == pygame.K_MINUS or event.key == pygame.K_KP_MINUS:
                zoom = max(0.2, zoom - zoom_step)
            elif event.key == pygame.K_UP:
                pan_y -= 50
            elif event.key == pygame.K_DOWN:
                pan_y += 50
            elif event.key == pygame.K_LEFT:
                pan_x -= 50
            elif event.key == pygame.K_RIGHT:
                pan_x += 50

    if data_index < len(data):
        entry = data[data_index]
        actions = entry['action']

        for action in actions:
            if action == "FORWARD":
                rad_angle = np.deg2rad(direction)
                new_x = x + step_size * np.sin(rad_angle)
                new_y = y - step_size * np.cos(rad_angle)
                # Draw line from last position to new position
                start = (int(x) + canvas_size // 2, int(y) + canvas_size // 2)
                end = (int(new_x) + canvas_size // 2, int(new_y) + canvas_size // 2)
                pygame.draw.line(path_surface, (0, 0, 255), start, end, 2)
                x, y = new_x, new_y
                path.append((x, y))
                dirty = True
            elif action == "RIGHT":
                direction = (direction + 90) % 360
            elif action == "LEFT":
                direction = (direction - 90) % 360

        data_index += 1

    if dirty:
        # Redraw virtual canvas only if something changed
        virtual_canvas.fill((107, 107, 107))
        virtual_canvas.blit(path_surface, (0, 0))
        pygame.draw.circle(
            virtual_canvas,
            (0, 255, 0),
            (int(path[0][0]) + canvas_size // 2, int(path[0][1]) + canvas_size // 2),
            8,
        )
        pygame.draw.circle(
            virtual_canvas,
            (255, 0, 0),
            (int(x) + canvas_size // 2, int(y) + canvas_size // 2),
            8,
        )

    # Display zoomed and panned view
    view_w = int(width / zoom)
    view_h = int(height / zoom)
    center_x = int(x) + canvas_size // 2 + pan_x
    center_y = int(y) + canvas_size // 2 + pan_y
    view_rect = pygame.Rect(center_x - view_w // 2, center_y - view_h // 2, view_w, view_h)
    zoomed_view = pygame.transform.scale(virtual_canvas.subsurface(view_rect), (width, height))
    screen.blit(zoomed_view, (0, 0))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
