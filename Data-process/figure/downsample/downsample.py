import os
from pathlib import Path
from PIL import Image, ImageFile
import concurrent.futures
import logging
from tqdm import tqdm
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='downsample.log'
)

size = (480, 480)
resolution = '360p'

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

def process_image(src_file, target_dir, size):
    """Process a single image - resize and save."""
    try:
        # Get base filename
        filename = os.path.basename(src_file)
        
        # Create target filename with prefix
        target_file = os.path.join(target_dir, resolution + "_" + filename)
        
        # Skip if target already exists
        if os.path.exists(target_file):
            return f"Skipped existing: {target_file}"
        
        # Open and resize image
        img = Image.open(src_file)
        resized_img = img.resize(size, Image.LANCZOS)
        
        # Save the result
        resized_img.save(target_file)
        
        return f"Processed: {filename}"
    
    except Exception as e:
        error_msg = f"Error processing {src_file}: {str(e)}"
        logging.error(error_msg)
        return error_msg

def main():
    # Set up paths
    os.chdir(r"E:\study-and-research\sunspot-with-sora")
    
    source_dir = Path("data/processed/figure/figure-origin")
    target_dir = Path('data/processed/figure/figure-downsample/' + resolution)
    
    # Create target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)
    
    # Find all PNG files
    image_files = list(source_dir.glob("*.png"))
    total_files = len(image_files)
    
    logging.info(f"Found {total_files} images to process")
    print(f"Found {total_files} images to process")
    
    # Set up parallel processing
    start_time = time.time()
    
    # Determine number of workers
    num_workers = min(os.cpu_count(), 8)  # Limit to 8 workers max
    if num_workers < 1:
        num_workers = 1
    
    processed_count = 0
    error_count = 0
    
    # Process files in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Create a list of future objects
        futures = {
            executor.submit(process_image, str(img_path), str(target_dir), size): img_path 
            for img_path in image_files
        }
        
        # Process results as they complete
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing images"):
            img_path = futures[future]
            try:
                result = future.result()
                if "Error" not in result:
                    processed_count += 1
                else:
                    error_count += 1
                    logging.error(f"Failed to process: {img_path}")
            except Exception as e:
                error_count += 1
                logging.error(f"Exception for {img_path}: {e}")
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    logging.info(f"Processing complete in {elapsed_time:.2f} seconds")
    logging.info(f"Processed {processed_count} images, {error_count} errors")
    print(f"Processing complete in {elapsed_time:.2f} seconds")
    print(f"Processed {processed_count} images, {error_count} errors")

if __name__ == "__main__":
    main()
