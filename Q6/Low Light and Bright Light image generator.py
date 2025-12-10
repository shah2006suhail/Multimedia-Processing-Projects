import cv2
import numpy as np

# Ask user for input file
img_path = input("Enter the image file path: ")

# Load image
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError("Could not load image. Check the file path.")

print("Image loaded successfully.")

# Convert to float for safe manipulation
img_float = img.astype(np.float32)

# ---- Generate LOW-LIGHT version ----
# Darken the image by multiplying with a factor < 1
low_light = img_float * 0.35     # reduce brightness to 35%
low_light = np.clip(low_light, 0, 255).astype(np.uint8)
cv2.imwrite("lowlight_generated.png", low_light)
print("Saved: lowlight_generated.png")

# ---- Generate BRIGHT-LIGHT version ----
# Brighten the image by multiplying with factor > 1
bright_light = img_float * 1.4   # increase brightness by 80%
bright_light = np.clip(bright_light, 0, 255).astype(np.uint8)
cv2.imwrite("brightlight_generated.png", bright_light)
print("Saved: brightlight_generated.png")

print("\nDone! Two synthetic lighting images created.")
