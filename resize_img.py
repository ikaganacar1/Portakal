from PIL import Image
import os

def resize_image_high_quality(input_path, output_path, new_width, new_height):
    if not os.path.exists(input_path):
        print(f"Error: The file {input_path} was not found.")
        return

    try:
        # Open the image
        with Image.open(input_path) as img:
            
            if img.mode != 'RGB':
                img = img.convert('RGB')

            print(f"Original size: {img.size}")
            resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Save the image optimizing for quality:
            # - quality=95: Best balance of file size and visual quality (100 can cause bloat).
            # - subsampling=0: Disables chroma subsampling, keeping colors as sharp as possible.
            resized_img.save(output_path, "JPEG", quality=95, subsampling=0)
            
            print(f"Successfully saved resized image to: {output_path}")
            print(f"New size: {(new_width, new_height)}")

    except Exception as e:
        print(f"An error occurred: {e}")

# === Example Usage ===
if __name__ == "__main__":
    INPUT_FILE = "img.jpg"
    OUTPUT_FILE = "img.jpg"
    
    TARGET_WIDTH = 256
    TARGET_HEIGHT = 256

    resize_image_high_quality(INPUT_FILE, OUTPUT_FILE, TARGET_WIDTH, TARGET_HEIGHT)
