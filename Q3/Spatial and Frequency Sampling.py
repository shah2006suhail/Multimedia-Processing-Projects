import cv2
import numpy as np
import os
import sys

# --- Configuration ---

# Get filename from command line if provided
input_filename_arg = sys.argv[1].strip() if len(sys.argv) > 1 else ""

# Ask user for input if no argument given
if input_filename_arg:
    img_filename = input_filename_arg
    print(f"Using command-line argument: {img_filename}")
else:
    img_filename = input("Enter the image filename (e.g., photo.png): ").strip()
    if not img_filename:
        print("\nError: No filename provided.")
        sys.exit(1)

# --- File existence check ---
if not os.path.isfile(img_filename):
    print(f"\nError: Could not find file '{img_filename}'.")
    sys.exit(1)

img = cv2.imread(img_filename, cv2.IMREAD_UNCHANGED)
if img is None:
    print(f"\nError: Could not read image file '{img_filename}'.")
    sys.exit(1)

# Convert to RGB correctly
if img.ndim == 2:  # grayscale
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
elif img.shape[2] == 4:  # BGRA
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
else:  # BGR
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Grayscale for frequency sampling
gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

# Sampling ratios
ratios = [2, 4, 8, 16]

save_path = os.getcwd()
root, ext = os.path.splitext(os.path.basename(img_filename))

print(f"Processing image: {img_filename}")
print(f"Saving output images in: {save_path}")

# ================================================================
# PART 1: SPATIAL SAMPLING (Reduce Resolution)
# ================================================================
print("\n--- Spatial Sampling (Saving Only Final Images) ---")

h, w = img_rgb.shape[:2]

for r in ratios:
    # Downsample every r-th pixel
    small = img_rgb[::r, ::r]

    # Upscale back with nearest neighbour
    resized = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    filename_cv = f"{root}_spatial_1_{r}.png"
    cv2.imwrite(os.path.join(save_path, filename_cv),
                cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))

    print(f"Saved: {filename_cv}")


# ================================================================
# PART 2: FREQUENCY SAMPLING (Reduce Frequencies)
# ================================================================
print("\n--- Frequency Sampling (Low-Pass Filter Only) ---")

# FFT
f = np.fft.fft2(gray)
fshift = np.fft.fftshift(f)

for r in ratios:
    fcopy = fshift.copy()

    # Centre
    cx, cy = fcopy.shape[0] // 2, fcopy.shape[1] // 2
    size_x, size_y = cx // r, cy // r

    # Low-pass mask
    mask = np.zeros(fshift.shape, dtype=bool)
    mask[cx - size_x : cx + size_x, cy - size_y : cy + size_y] = True

    fcopy[~mask] = 0

    ishift = np.fft.ifftshift(fcopy)
    sampled = np.abs(np.fft.ifft2(ishift))

    # Normalize to 8-bit
    sampled_norm = cv2.normalize(sampled, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    filename_cv = f"{root}_freq_1_{r}.png"
    cv2.imwrite(os.path.join(save_path, filename_cv), sampled_norm)

    print(f"Saved: {filename_cv}")

print("\nAll output images saved successfully.")
