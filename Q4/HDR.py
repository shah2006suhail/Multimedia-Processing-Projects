import cv2
import numpy as np
import os

class HDRProcessor:
    def __init__(self):
        self.response_function = None
        self.hdr_image = None
    
    def load_images(self, image_folder, exposure_times):
        img_list = []
        print(f"Loading images from {image_folder}...")
        
        # Sorts files: img1.jpg, img2.jpg, img3.jpg
        filenames = sorted([f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        
        for filename in filenames:
            img = cv2.imread(os.path.join(image_folder, filename))
            if img is not None:
                img_list.append(img)
                print(f" -> Loaded {filename}")
        
        if len(img_list) != len(exposure_times):
            print(f"\nERROR: Found {len(img_list)} images, but you listed {len(exposure_times)} exposure times.")
            print("Ensure your folder has exactly 3 images named img1, img2, img3.")
            raise ValueError("Image count mismatch.")
            
        return img_list, np.array(exposure_times, dtype=np.float32)

    def process_hdr(self, images, times):
        print("\nStep 1: Estimating Camera Response Function...")
        calibrate = cv2.createCalibrateDebevec()
        response_function = calibrate.process(images, times)
        
        print("Step 2: Merging to HDR...")
        merge_debevec = cv2.createMergeDebevec()
        hdr_image = merge_debevec.process(images, times, response_function)
        return hdr_image

    def tone_map(self, hdr_image):
        print("Step 3: Tone Mapping...")
        tonemap = cv2.createTonemapDrago(1.0, 1.0)
        ldr = tonemap.process(hdr_image)
        
        # --- FIX: Handle NaN/Errors from non-RAW images ---
        ldr = ldr * 255
        ldr = np.nan_to_num(ldr, nan=0.0, posinf=255.0, neginf=0.0)
        return np.clip(ldr, 0, 255).astype('uint8')

if __name__ == "__main__":
    # --- MANUAL CONFIGURATION ---
    # Since we can't read the files, we estimate standard bracket times:
    exposure_times = [0.35,1.0,1.4] 
    
    folder = 'exposure_sequence'
    
    try:
        processor = HDRProcessor()
        images, times = processor.load_images(folder, exposure_times)
        
        hdr_data = processor.process_hdr(images, times)
        final_img = processor.tone_map(hdr_data)
        
        cv2.imwrite('output_manual_fix.jpg', final_img)
        print("\nSUCCESS! Saved 'output_manual_fix.jpg'")
        
    except Exception as e:
        print(f"\nERROR: {e}")
    
