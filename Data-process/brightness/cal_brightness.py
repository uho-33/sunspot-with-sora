import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
import datetime
import concurrent.futures
from tqdm import tqdm  # Changed from tqdm.notebook to regular tqdm
import os
import pandas as pd
import re
import multiprocessing
import logging

def extract_timestamp(filename):
    """
    Extract timestamp from the filename.
    Assumes filename format contains date/time information.
    Modify the regex pattern based on your actual filename format.
    """
    # Example pattern: looking for something like YYYY-MM-DD_HH-MM-SS
    # Convert Path object to string if needed
    filename_str = str(filename)
    pattern = r'hmi\.ic_720s\.(\d{8}_\d{6})_TAI\.3\.continuum_nd\.h5'
    match = re.search(pattern, filename_str)
    
    if match:
        # Convert matched string to datetime
        dt_str = match.group(1)
        # Format: YYYYMMDD_HHMMSS to YYYY-MM-DD HH:MM:SS
        formatted_dt = f"{dt_str[:4]}-{dt_str[4:6]}-{dt_str[6:8]} {dt_str[9:11]}:{dt_str[11:13]}:{dt_str[13:15]}"
        return datetime.datetime.strptime(formatted_dt, "%Y-%m-%d %H:%M:%S")
    
    # Fallback: use file modification time if pattern doesn't match
    file_stat = os.stat(filename)
    return datetime.datetime.fromtimestamp(file_stat.st_mtime)

def process_file(file_path):
    """Process a single file to extract timestamp and calculate brightness."""
    try:
        with h5py.File(file_path, 'r') as hf:
            # Get data and calculate brightness
            data = hf['data'][:]
            brightness = np.nansum(data)
            
            # Try to get timestamp from file metadata if available
            timestamp = None
            if 'timestamp' in hf.attrs:
                timestamp = hf.attrs['timestamp']
            else:
                # Extract from filename
                timestamp = extract_timestamp(file_path)
                
            return timestamp, brightness
    except Exception as e:
        # Enhanced error logging with full path and error details
        error_msg = f"Error processing {file_path}: {e}"
        logging.error(error_msg)
        # Return a tuple with None and error information for tracking
        return None, (file_path, str(e))

def process_files_in_parallel(h5_files):
    """Process all files in parallel and return results and errors."""
    print("Calculating total brightness for each file in parallel...")
    start_time = time.time()

    # Define number of workers based on CPU cores
    num_workers = os.cpu_count() - 1  # Leave one core free
    if num_workers < 1:
        num_workers = 1

    results = []
    corrupted_files = []
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(process_file, file_path): file_path for file_path in h5_files}
        
        # Process results as they complete
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            file_path = futures[future]
            try:
                result = future.result()
                if isinstance(result, tuple) and len(result) == 2:
                    if result[0] is not None:
                        results.append(result)  # This is a successful result
                    elif result[1] is not None:
                        corrupted_files.append(result[1])  # This is error information
            except Exception as e:
                logging.error(f"Error processing {file_path}: {e}")
                corrupted_files.append((file_path, str(e)))

    end_time = time.time()
    print(f"Parallel processing completed in {end_time - start_time:.2f} seconds")
    
    # Log summary of corrupted files
    if corrupted_files:
        logging.warning(f"Found {len(corrupted_files)} corrupted files")
        
    return results, corrupted_files

def analyze_results(results, save_path, prefix=""):
    """Analyze and save results."""
    # Convert results to a DataFrame for easier manipulation
    brightness_df = pd.DataFrame(results, columns=['timestamp', 'brightness'])
    brightness_df = brightness_df.sort_values('timestamp')

    # save to numpy format
    np_path = save_path + prefix + "total_brightness_timeseries.npz"
    np.savez(np_path, 
             timestamps=brightness_df['timestamp'].astype('datetime64[ns]').values,
             brightness=brightness_df['brightness'].values)
    print(f"Saved numpy array to {np_path}")

    # Analyze results
    brightness = brightness_df['brightness']

    mean = np.nanmean(brightness)
    std = np.nanstd(brightness)

    stats_path = save_path + prefix + "brightness_stats.npz"
    np.savez(stats_path, mean=mean, std=std)
    print(f"Saved mean and std to {stats_path}")

    return brightness_df


def main():
    """Main function to run the entire analysis."""
    os.chdir(r"E:\study-and-research\sunspot-with-sora")
    
    # Directory containing the processed h5 files
    data_dir = Path("data/processed/Ic_720s_normalize_distance")
    
    prefix = data_dir.parts[-1]
    
    save_path = "data/processed/"
    
    # Get all h5 files
    h5_files = list(data_dir.glob("*.h5"))
    print(f"Found {len(h5_files)} h5 files")
    
    # Process files and get errors
    results, corrupted_files = process_files_in_parallel(h5_files)
    
    # Save corrupted files list to a file
    if corrupted_files:
        error_file = Path(save_path) / f"{prefix}corrupted_files.csv"
        with open(error_file, 'w') as f:
            f.write("file_path,error\n")  # CSV header
            for file_path, error in corrupted_files:
                f.write(f"{file_path},{error.replace(',', ';')}\n")
        
        print(f"Saved {len(corrupted_files)} corrupted files to {error_file}")
        
    analyze_results(results, save_path, prefix)
    
if __name__ == "__main__":
    # This is required for Windows to prevent recursive process spawning
    log_dir = "Data-process"
    log_file = Path(log_dir) / "nb.log"
    logging.basicConfig(filename=log_file, level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    multiprocessing.freeze_support()
    main()