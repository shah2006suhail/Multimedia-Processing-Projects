import cv2
import numpy as np
import os

print("=== Bit-Plane Slicing Program ===\n")

# Ask user for input files
low_img_path = input("Enter the path for your LOW-LIGHT image: ")
bright_img_path = input("Enter the path for your BRIGHT-LIGHT image: ")

# Output folder
output_dir = "bitplane_outputs"
os.makedirs(output_dir, exist_ok=True)

def process_image(image_path, prefix):
    print(f"\nProcessing image: {image_path}")
    
    # Step 1: Load image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: Cannot load image. Check the file path.")
        return
    print("  Loaded image successfully.")

    # Save original for reference
    cv2.imwrite(f"{output_dir}/{prefix}_original.png", img)

    # Step 2: Extract bit-planes 0..7
    print("  Extracting bit-planes 0 to 7...")
    for b in range(8):
        plane = ((img >> b) & 1) * 255
        cv2.imwrite(f"{output_dir}/{prefix}_bitplane_{b}.png", plane)
        print(f"    Saved bit-plane {b}")

    # Step 3: Reconstruct image from bit0, bit1, bit2 (lowest 3 bits)
    print("  Reconstructing using lowest 3 bit-planes...")
    bit0 = (img & 1)
    bit1 = ((img >> 1) & 1)
    bit2 = ((img >> 2) & 1)

    reconstructed = bit0*1 + bit1*2 + bit2*4
    reconstructed = reconstructed.astype(np.uint8)

    # Scale reconstructed for visibility (0..7 → 0..255)
    reconstructed_vis = (reconstructed * 32).astype(np.uint8)
    cv2.imwrite(f"{output_dir}/{prefix}_reconstructed.png", reconstructed_vis)
    print("    Saved reconstructed image.")

    # Step 4: Compute difference image
    print("  Computing difference (original - reconstructed)...")
    difference = cv2.absdiff(img, reconstructed_vis)
    cv2.imwrite(f"{output_dir}/{prefix}_difference.png", difference)
    print("    Saved difference image.")

    print(f"Finished processing {prefix}.\n")

# Process both images
process_image(low_img_path, "lowlight")
process_image(bright_img_path, "brightlight")

print("\nAll results saved in folder:", output_dir)
