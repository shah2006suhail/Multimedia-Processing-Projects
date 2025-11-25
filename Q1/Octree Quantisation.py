from PIL import Image
import os

# Ask user for input file
input_path = input("Enter the image filename (e.g., photo.png): ").strip()

if not os.path.isfile(input_path):
    raise FileNotFoundError("The specified file does not exist.")

# Ask user for number of colours
colors = int(input("Enter number of colours (1–256): ").strip())
colors = max(1, min(colors, 256))  # clamp to safe range
print("Using", colors, "colours.")

# Build output filename safely
root, ext = os.path.splitext(input_path)
output_path = f"{root}_octree_{colors}colors{ext}"

# Load image
img = Image.open(input_path)

# Handle alpha if present
has_alpha = (img.mode == "RGBA" or img.mode == "LA")

if has_alpha:
    alpha = img.getchannel("A")
    rgb = img.convert("RGB")
else:
    alpha = None
    rgb = img.convert("RGB")

# Apply Octree quantisation with fallback
try:
    quant = rgb.quantize(colors=colors, method=Image.FASTOCTREE)
except Exception:
    quant = rgb.quantize(colors=colors, method=Image.MAXCOVERAGE)

# Convert back to plain RGB
quant_rgb = quant.convert("RGB")

# Reattach alpha if needed
if alpha is not None:
    r, g, b = quant_rgb.split()
    result = Image.merge("RGBA", (r, g, b, alpha))
else:
    result = quant_rgb

# Save output
result.save(output_path)
print("Saved:", output_path)
