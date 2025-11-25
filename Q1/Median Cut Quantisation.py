from PIL import Image
import os

# Ask user for input file
input_path = input("Enter the image filename (e.g., photo.png): ").strip()

if not os.path.isfile(input_path):
    raise FileNotFoundError("The specified file does not exist.")

# Ask for number of colours
colors = int(input("Enter number of colours (1–256): ").strip())
colors = max(1, min(colors, 256))  # safety clamp
print("Using", colors, "colours.")

# Build output filename
root, ext = os.path.splitext(input_path)
output_path = f"{root}_mediancut_{colors}colors{ext}"

# Load image
img = Image.open(input_path)

# Check if the image has an alpha channel
has_alpha = (img.mode == "RGBA" or img.mode == "LA")

if has_alpha:
    img = img.convert("RGBA")
    alpha = img.getchannel("A")
    rgb = img.convert("RGB")
else:
    rgb = img.convert("RGB")
    alpha = None

# Apply Median-Cut quantisation
quant = rgb.quantize(colors=colors, method=Image.MEDIANCUT)

# Convert back to RGB
quant_rgb = quant.convert("RGB")

# If alpha existed, reattach it
if alpha:
    r, g, b = quant_rgb.split()
    result = Image.merge("RGBA", (r, g, b, alpha))
else:
    result = quant_rgb

# Save output
result.save(output_path)
print("Saved:", output_path)
