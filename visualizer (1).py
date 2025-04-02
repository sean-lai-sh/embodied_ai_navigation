import os
import json
import cv2

# Set the path to your images and JSON
save_dir = "data/images"  # <-- Change this
json_path = "data/data_info.json"

# Load metadata
with open(json_path, "r") as f:
    data = json.load(f)

# Helper function to display image with text
def show_image(index):
    if index < 0 or index >= len(data):
        print("Index out of bounds.")
        return index

    entry = data[index]
    image_path = os.path.join(save_dir, entry["image"])
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return index

    img = cv2.imread(image_path)
    text = f"Step: {entry['step']} | Action: {', '.join(entry['action'])}"
    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Data Viewer", img)
    return index

def run_viewer():
    index = 0
    index = show_image(index)

    while True:
        key = cv2.waitKey(0)

        if key == ord('q'):
            break
        elif key == ord('a'):  # Previous
            index = max(0, index - 1)
        elif key == ord('d'):  # Next
            index = min(len(data) - 1, index + 1)

        index = show_image(index)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_viewer()