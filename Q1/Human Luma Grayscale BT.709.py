from PIL import Image
import os

# Ask user for the image filename
input_path = input("Enter the image filename (e.g., photo.png): ").strip()

if not os.path.isfile(input_path):
    raise FileNotFoundError("The specified file does not exist.")

# Build output filename safely
root, ext = os.path.splitext(input_path)
output_path = f"{root}_luma_gray{ext}"

# Load image
img = Image.open(input_path)

# Handle images with or without alpha
has_alpha = (img.mode == "RGBA" or img.mode == "LA")

if has_alpha:
    # Keep alpha separately
    alpha = img.getchannel("A")
    rgb = img.convert("RGB")
else:
    alpha = None
    rgb = img.convert("RGB")

# Prepare pixel access
pixels = rgb.load()

# BT.709 luma weights
for y in range(rgb.height):
    for x in range(rgb.width):
        r, g, b = pixels[x, y]
        y_luma = int(0.2126 * r + 0.7152 * g + 0.0722 * b)
        pixels[x, y] = (y_luma, y_luma, y_luma)

# Reattach alpha if present
if alpha is not None:
    r, g, b = rgb.split()
    result = Image.merge("RGBA", (r, g, b, alpha))
else:
    result = rgb

# Save result
result.save(output_path)
print("Saved:", output_path)
