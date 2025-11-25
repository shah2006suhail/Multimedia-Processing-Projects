from PIL import Image

# Ask user for the image filename
input_path = input("Enter the image filename (e.g., photo.png): ")

# Build output filename
dot = input_path.rfind(".")
output_path = input_path[:dot] + "_luma_gray" + input_path[dot:]

# Convert to grayscale using human-vision BT.709 luma
img = Image.open(input_path).convert("RGB")
pixels = img.load()

for y in range(img.height):
    for x in range(img.width):
        r, g, b = pixels[x, y]
        y_luma = int(0.2126*r + 0.7152*g + 0.0722*b)
        pixels[x, y] = (y_luma, y_luma, y_luma)

# Save result
img.save(output_path)
print("Saved:", output_path)
