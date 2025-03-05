import h5py
import numpy as np
import os
import logging
from pathlib import Path
import concurrent.futures
from tqdm import tqdm
import time
from PIL import Image
import gc

# Configure logging
log_dir = Path("Data-process/figure")
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / "figure_norm.log"
logging.basicConfig(filename=log_file, level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')

def count_valid_points(file_path):
    """Count the number of valid (non-NaN) data points in an h5 file."""
    try:
        with h5py.File(file_path, 'r') as hf:
            data = hf['data'][:]
            valid_count = np.count_nonzero(~np.isnan(data))
            return valid_count
    except Exception as e:
        logging.error(f"Error counting valid points in {file_path}: {e}")
        return None

def transform_to_image(file_path, output_dir, m, sigma):
    """Transform data to image using the given transformation parameters."""
    try:
        # Extract base filename
        base_name = os.path.basename(file_path)
        output_name = os.path.splitext(base_name)[0] + ".png"
        output_path = os.path.join(output_dir, output_name)
        
        # Skip if output already exists
        if os.path.exists(output_path):
            return True, f"Skipped existing: {output_name}"
        
        # Read data
        with h5py.File(file_path, 'r') as hf:
            data = hf['data'][:]
            
        # Transform data to image
        nan_replaced_data = np.nan_to_num(data, nan=0)
        
        # Apply transformation
        fig_arr = np.clip((nan_replaced_data - (m - 3 * sigma)) / (6 * sigma) * 255, 0, 255).astype(np.uint8)
        
        # Create and save image
        image = Image.fromarray(fig_arr)
        image.save(output_path)
        
        # Free memory
        del data, nan_replaced_data, fig_arr, image
        gc.collect()
        
        return True, f"Processed: {output_name}"
    
    except Exception as e:
        logging.error(f"Error processing {file_path}: {e}")
        return False, f"Error: {file_path} - {str(e)}"

def main():
    """Main function to process all h5 files."""
    try:
        # Set working directory
        os.chdir(r"E:\study-and-research\sunspot-with-sora")
        
        # Directories
        input_dir = Path("data/processed/Ic_nolimbdark_720s_normalize_distance")
        output_dir = Path("data/processed/figure")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all h5 files
        h5_files = list(input_dir.glob("*.h5"))
        total_files = len(h5_files)
        logging.info(f"Found {total_files} h5 files to process")
        print(f"Found {total_files} h5 files to process")
        
        # Calculate mean number of valid points (n)
        print("Calculating mean number of valid points...")
        logging.info("Calculating mean number of valid points...")
        
        start_time = time.time()

        # Read brightness statistics
        brightness_stats_path = Path("data/processed/Ic_720s_normalize_distance_brightness_stats.npz")
        
        stats = np.load(brightness_stats_path)
        mean_brightness = stats['mean']
        std_brightness = stats['std']
        
        # Calculate normalization parameters
        m = 1
        sigma = std_brightness / mean_brightness
        
        logging.info(f"Normalization parameters: m={m}, sigma={sigma}")
        print(f"Normalization parameters: m={m}, sigma={sigma}")
        
        # Process all files and transform to images
        print("Processing files and generating images...")
        logging.info("Processing files and generating images...")
        
        # Define number of workers based on CPU cores
        num_workers = os.cpu_count() - 1  # Leave one core free
        if num_workers < 1:
            num_workers = 1
        
        processed_count = 0
        error_count = 0
        
        # Process files in parallel
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Process files in batches to prevent memory issues
            batch_size = 50
            for i in range(0, len(h5_files), batch_size):
                batch = h5_files[i:i+batch_size]
                
                # Submit batch for processing
                futures = {
                    executor.submit(transform_to_image, str(file_path), str(output_dir), m, sigma): file_path
                    for file_path in batch
                }
                
                # Process results as they complete
                for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"Batch {i//batch_size + 1}/{(len(h5_files)-1)//batch_size + 1}"):
                    file_path = futures[future]
                    try:
                        success, message = future.result()
                        if success:
                            processed_count += 1
                        else:
                            error_count += 1
                    except Exception as e:
                        error_count += 1
                        logging.error(f"Error processing {file_path}: {e}")
                
                # Force garbage collection between batches
                gc.collect()
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        logging.info(f"Processing complete. Time: {elapsed_time:.2f} seconds")
        logging.info(f"Processed {processed_count} files, {error_count} errors")
        print(f"Processing complete. Time: {elapsed_time:.2f} seconds")
        print(f"Processed {processed_count} files, {error_count} errors")
        
    except Exception as e:
        logging.error(f"Error in main function: {e}")
        print(f"Error in main function: {e}")

if __name__ == "__main__":
    main()
