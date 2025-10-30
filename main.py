#!/usr/bin/env python3
import sys
import time
import subprocess
import shutil
import cv2
import numpy as np
from PIL import Image

# -----------------------------
# ydotool helpers
# -----------------------------
def check_ydotool():
    return shutil.which("ydotool") is not None

def ydotool_move_absolute(x, y):
    subprocess.run(["ydotool", "mousemove", "--absolute", "--x", str(x), "--y", str(y)], check=True)

def ydotool_click_left():
    subprocess.run(["ydotool", "click", "0xC0"], check=True)

# -----------------------------
# Image processing
# -----------------------------
def load_contours(image_path, target_width):
    img = Image.open(image_path).convert("L")
    w, h = img.size
    scale_factor = target_width / w
    new_w = target_width
    new_h = int(h * scale_factor)
    img = img.resize((new_w, new_h), Image.LANCZOS)

    arr = np.array(img)
    arr = cv2.GaussianBlur(arr, (3,3), 0)

    # Auto Canny edge detection
    v = np.median(arr)
    lower = int(max(0, 0.66 * v))
    upper = int(min(255, 1.33 * v))
    edges = cv2.Canny(arr, lower, upper)

    # Dilate to thicken lines
    kernel = np.ones((2,2), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    sampled_contours = []
    for c in contours:
        pts = c.reshape(-1,2)
        if len(pts) < 5:
            continue
        idx = np.linspace(0, len(pts)-1, min(300, len(pts))).astype(int)
        sampled_contours.append([tuple(pts[i]) for i in idx])

    return sampled_contours, new_w, new_h

# -----------------------------
# Draw contours using absolute positions
# -----------------------------
def draw_contours_absolute(contours, start_x=100, start_y=100):
    """
    Draw contours using absolute positions.
    start_x, start_y: the absolute top-left of where to start drawing
    """
    for contour in contours:
        if not contour:
            continue

        # Move to first point
        first_x = start_x + contour[0][0]
        first_y = start_y + contour[0][1]
        ydotool_move_absolute(first_x, first_y)
        ydotool_click_left()

        # Draw the rest of the contour
        for x, y in contour[1:]:
            abs_x = start_x + x
            abs_y = start_y + y
            ydotool_move_absolute(abs_x, abs_y)
            ydotool_click_left()
            time.sleep(0.002)

# -----------------------------
# Main
# -----------------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python3 draw_absolute.py path/to/image.png")
        sys.exit(1)

    if not check_ydotool():
        print("❌ ydotool not found. Please install it first.")
        sys.exit(1)

    image_path = sys.argv[1]
    target_width = int(input("Enter desired image width in pixels (e.g., 100): "))

    contours, img_w, img_h = load_contours(image_path, target_width)
    print(f"✅ Loaded {image_path}, scaled to {img_w}x{img_h}, found {len(contours)} contours.")

    input("Prepare your drawing window and press Enter...")
    print("⏳ Waiting 5 seconds to switch to your drawing window...")
    time.sleep(5)

    print("🎨 Starting drawing with absolute positions...")
    draw_contours_absolute(contours, start_x=400, start_y=600)

    print("✅ Done drawing!")

if __name__ == "__main__":
    main()

