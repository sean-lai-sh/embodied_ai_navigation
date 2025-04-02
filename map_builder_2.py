import json
import pygame
import numpy as np
import os

# --- Setup ---
pygame.init()
WIDTH, HEIGHT = 800, 800
CANVAS_SIZE = 5000
STEP_SIZE = 20
canvas = pygame.Surface((CANVAS_SIZE, CANVAS_SIZE))
canvas.fill((107, 107, 107))
path_surface = pygame.Surface((CANVAS_SIZE, CANVAS_SIZE), pygame.SRCALPHA)

# --- Load Data ---
with open("data/data_info.json", "r") as f:
    data = json.load(f)

image_folder = "data/Images"
x, y = 0, 0
direction = 0
position_to_image = {}

# --- Movement Logic ---
def move_forward(x, y, angle):
    rad = np.deg2rad(angle)
    return (
        x + STEP_SIZE * np.sin(rad),
        y - STEP_SIZE * np.cos(rad)
    )

# --- Process Exploration Log ---
for entry in data:
    for action in entry["action"]:
        if action == "FORWARD":
            new_x, new_y = move_forward(x, y, direction)
            start = (int(x) + CANVAS_SIZE // 2, int(y) + CANVAS_SIZE // 2)
            end = (int(new_x) + CANVAS_SIZE // 2, int(new_y) + CANVAS_SIZE // 2)
            pygame.draw.line(path_surface, (0, 0, 255), start, end, 2)
            x, y = new_x, new_y
        elif action == "LEFT":
            direction = (direction - 90) % 360
        elif action == "RIGHT":
            direction = (direction + 90) % 360

    key = (int(x), int(y))
    position_to_image[key] = os.path.join(image_folder, entry["image"])

# --- Save JSON with string keys ---
position_to_image_str = {f"{k[0]},{k[1]}": v for k, v in position_to_image.items()}
with open("map_dict.json", "w") as f:
    json.dump(position_to_image_str, f, indent=2)

# --- Save Maze Visualization ---
canvas.blit(path_surface, (0, 0))
pygame.image.save(canvas, "maze_map.png")

print("✅ Map built successfully.")
print("🗺️  Saved visual map as maze_map.png")
print("🗂️  Saved image-position map as map_dict.json")
