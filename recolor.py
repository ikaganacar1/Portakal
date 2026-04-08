import cv2
import numpy as np
import os
import random

def recolor_orange(input_path, output_folder, num_variants=10):
    # 1. Load the image
    img = cv2.imread(input_path)
    if img is None:
        print("Error: Could not find image.")
        return

    # Create output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 2. Convert to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # 3. Create a mask to ignore the white background
    # Adjust the 250 threshold if your 'white' isn't perfectly pure
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = gray < 250 

    for i in range(num_variants):
        # 4. Generate a random hue shift (0-179 in OpenCV)
        random_hue = random.randint(0, 179)
        
        # Apply shift only to the masked area (the fruit)
        new_h = h.copy()
        new_h[mask] = (new_h[mask].astype(int) + random_hue) % 180
        
        # 5. Merge back and convert to BGR
        new_hsv = cv2.merge([new_h.astype(np.uint8), s, v])
        result = cv2.cvtColor(new_hsv, cv2.COLOR_HSV2BGR)

        # 6. Save the file
        output_filename = f"orange_variant_{i:03d}.jpg"
        cv2.imwrite(os.path.join(output_folder, output_filename), result)
        print(f"Saved: {output_filename}")

# --- Configuration ---
input_image = "img.jpg"  # Path to your file
target_dir = "colored_oranges"
recolor_orange(input_image, target_dir, num_variants=256)
