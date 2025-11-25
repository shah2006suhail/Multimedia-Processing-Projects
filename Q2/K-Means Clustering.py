import cv2
import numpy as np
from sklearn.cluster import KMeans
import os
import sys

def run_color_quantization():
    """
    Performs color quantization on an image using K-Means clustering.
    Handles BGR/RGB conversion necessary when mixing OpenCV and Scikit-learn.
    """

    # --- 1. User Input and Validation ---
    
    # Ask user for the image filename/path
    input_path = input("Enter the image filename (e.g., photo.png): ").strip()

    if not os.path.isfile(input_path):
        # Use sys.exit or raise an error to stop execution clearly
        print(f"Error: The specified file '{input_path}' does not exist.", file=sys.stderr)
        return

    # Ask user for K value
    try:
        k = int(input("Enter number of colours (1–256): ").strip())
    except ValueError:
        print("Error: Invalid K value. Please enter a whole number.", file=sys.stderr)
        return

    # Safety clamp K value between 1 and 256
    k = max(1, min(k, 256))
    print(f"Using {k} colours.")

    # --- 2. Load and Prepare Image ---
    
    print(f"Reading image: {input_path} ...")
    
    # Load image (OpenCV loads in BGR format)
    img_bgr = cv2.imread(input_path)

    if img_bgr is None:
        print("Error: Unable to load image. Check the file format.", file=sys.stderr)
        return

    # **FIX 1: Convert BGR to RGB**
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Get original dimensions
    h, w, c = img_rgb.shape

    # Prepare pixel data: flatten the image and convert to float32
    pixels = img_rgb.reshape((-1, 3)).astype(np.float32)

    # --- 3. Run K-Means Clustering ---
    
    print("Running K-Means...")
    # The K-Means algorithm groups the pixels in the 3D RGB color space. 
    # The cluster centers are the final, representative colors. 
    # 
    
    # n_init='auto' is the current recommended setting
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    
    # Run fit_predict to train the model and get the cluster index for every pixel
    labels = kmeans.fit_predict(pixels)

    # Extract cluster centres and convert back to 8-bit integers (0-255)
    centers_rgb = np.uint8(kmeans.cluster_centers_)
    
    # --- 4. Reconstruct and Save Image ---

    # Reconstruct the image by mapping labels to the new center colors
    # The segmented image data is currently in RGB
    segmented_data_rgb = centers_rgb[labels]
    output_img_rgb = segmented_data_rgb.reshape((h, w, c))
    
    # **FIX 2: Convert RGB back to BGR before saving**
    # cv2.imwrite expects BGR format
    final_output_img_bgr = cv2.cvtColor(output_img_rgb, cv2.COLOR_RGB2BGR)

    # Build output filename
    root, ext = os.path.splitext(input_path)
    output_path = f"{root}_kmeans_{k}colors{ext}"

    # Save output file
    cv2.imwrite(output_path, final_output_img_bgr)
    print(f"\nSUCCESS: Quantized image saved to: {output_path}")
    print("Done!")

if __name__ == "__main__":
    run_color_quantization()