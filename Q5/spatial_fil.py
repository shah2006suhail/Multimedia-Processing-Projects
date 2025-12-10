import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# --------------------------------------------------
# 1. LOAD IMAGE USING YOUR PATH
# --------------------------------------------------
path = r"C:\Users\mohne\OneDrive\Desktop\multi\Torgya - Arunachal Festival.jpg"
img = cv2.imread(path)
if img is None:
    raise FileNotFoundError(f"Failed to load image. Check file path:\n{path}")

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Folder to save results (same as image folder)
save_path = r"C:\Users\mohne\OneDrive\Desktop\multi"

# --------------------------------------------------
# 2. BOX FILTER FUNCTION
# --------------------------------------------------
def box_filter(image, k, normalize=True):
    kernel = np.ones((k, k), dtype=np.float32)
    if normalize:
        kernel = kernel / (k * k)
    return cv2.filter2D(image, -1, kernel)

# Compute filters
box5_norm     = box_filter(img, 5, True)
box5_non_norm = box_filter(img, 5, False)

box20_norm     = box_filter(img, 20, True)
box20_non_norm = box_filter(img, 20, False)

# --------------------------------------------------
# 3. GAUSSIAN FILTER SIZE
# --------------------------------------------------
sigma = 3.0
gaussian_size = int(6 * sigma + 1)
print("Sigma =", sigma, " => Gaussian kernel size =", gaussian_size)

# --------------------------------------------------
# 4. SEPARABLE GAUSSIAN FILTERS
# --------------------------------------------------
def gaussian_1d(sigma, size):
    ax = np.arange(-(size//2), size//2 + 1)
    kernel = np.exp(-0.5 * (ax / sigma)**2)
    return kernel

g = gaussian_1d(sigma, gaussian_size)
g_norm = g / np.sum(g)

g2d      = np.outer(g, g)
g2d_norm = np.outer(g_norm, g_norm)

gaussian_sep      = cv2.filter2D(img, -1, g2d)
gaussian_sep_norm = cv2.filter2D(img, -1, g2d_norm)

# --------------------------------------------------
# 5. SAVE ALL OUTPUT IMAGES (NO POP-UP)
# --------------------------------------------------
def save(name, image):
    cv2.imwrite(
        os.path.join(save_path, name),
        cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    )

save("original.jpg", img)
save("box5_normalized.jpg", box5_norm)
save("box5_not_normalized.jpg", box5_non_norm)
save("box20_normalized.jpg", box20_norm)
save("box20_not_normalized.jpg", box20_non_norm)
save("gaussian_separable.jpg", gaussian_sep)
save("gaussian_separable_normalized.jpg", gaussian_sep_norm)

print("\nAll images saved successfully to:\n", save_path, "\n")
