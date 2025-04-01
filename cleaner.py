import json

# Load raw data
with open("./data/data_info.json") as f:
    data = json.load(f)

cleaned = []
prev_action = None
action_buffer = []
frame_interval = 3  # Keep 1 of every 3 in long stretches

for i, entry in enumerate(data):
    curr_action = entry['action'][0]

    # If changing action, flush buffer
    if curr_action != prev_action and action_buffer:
        # Keep last frame of previous action
        cleaned.append(action_buffer[-1])
        action_buffer = []

    # For IDLE: only keep 1 in a row
    if curr_action == "IDLE":
        if prev_action != "IDLE":
            cleaned.append(entry)

    else:
        action_buffer.append(entry)
        # Optionally keep every nth image
        if len(action_buffer) % frame_interval == 0:
            cleaned.append(entry)

    prev_action = curr_action

# Save cleaned dataset
with open("./data/data_info_cleaned.json", "w") as f:
    json.dump(cleaned, f, indent=4)
