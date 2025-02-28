import astropy.units as u
import sunpy.map
import numpy as np
import os
import h5py
import matplotlib.pyplot as plt
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from glob import glob
import logging
from datetime import datetime
import gc  # Add garbage collection
import time  # For timing stats

# Configure logging
log_dir = "Data-process"
log_file = Path(log_dir) / "processing.log"
logging.basicConfig(filename=log_file, level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def normalize_distance(data_map):
    """Normalize the data based on distance to Sun in AU."""
    data_out = data_map.data * (data_map.dsun.to(u.AU)/(1 * u.AU).value)**2
    return data_out

def save_as_hdf5(data, output_path):
    """Save normalized data as HDF5 file."""
    with h5py.File(output_path, 'w') as hf:
        hf.create_dataset('data', data=data, compression="gzip", compression_opts=9)

def is_processed(input_path, output_dir):
    """Check if a file has already been processed."""
    base_name = Path(input_path).stem
    hdf5_path = Path(output_dir) / f"{base_name}_nd.h5"
    return hdf5_path.exists()

def get_output_path(input_path, output_dir):
    """Get the output HDF5 path for a given input file."""
    base_name = Path(input_path).stem
    return Path(output_dir) / f"{base_name}_nd.h5"

def process_fits_file(input_path, output_dir, allow_errors=False):
    """Process a FITS file and save normalized data as HDF5."""
    try:
        # Load and process data
        data_map = sunpy.map.Map(input_path)
        normalized_data = normalize_distance(data_map)
        
        # Generate output filename and save
        hdf5_path = get_output_path(input_path, output_dir)
        save_as_hdf5(normalized_data, hdf5_path)
        
        # Explicitly clear references to free memory
        del data_map
        del normalized_data
        gc.collect()  # Force garbage collection
        
        # logging.info(f"Saved HDF5 file: {hdf5_path}")
        
        return True
        
    except Exception as e:
        logging.error(f"Error processing file {input_path}: {e}")
        if not allow_errors:
            raise
        return False


if __name__ == "__main__":
    start_time = time.time()
    os.chdir(r"E:\study-and-research\sunspot-with-sora")

    # Get all FITS files in the input directory
    input_dir = Path(r"data\origin\Ic_nolimbdark_720s")
    output_dir = Path(r"data\processed\Ic_nolimbdark_720s_normalize_distance")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Pre-filter files to process
    fits_files = list(input_dir.glob("*.fits"))
    total_files = len(fits_files)
    
    # Pre-check which files need processing
    files_to_process = []
    skipped_files = 0
    
    logging.info(f"Checking {total_files} files for processing status...")
    
    for file in fits_files:
        if is_processed(file, output_dir):
            skipped_files += 1
        else:
            files_to_process.append(file)
    
    need_processing = len(files_to_process)
    logging.info(f"Found {need_processing} files to process, {skipped_files} already processed")
    
    # Counter for processed files
    processed_files = 0
    error_files = 0

    # Process files in parallel with optimal workers
    max_workers = 10
    
    if files_to_process:
        logging.info(f"Starting processing with {max_workers} workers")
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Process files in batches to prevent memory buildup
            batch_size = 10
            for i in range(0, len(files_to_process), batch_size):
                batch = files_to_process[i:i+batch_size]
                # logging.info(f"Processing batch {i//batch_size + 1} of {(len(files_to_process)-1)//batch_size + 1} ({len(batch)} files)")
                
                # Submit batch for processing
                futures = [
                    executor.submit(process_fits_file, str(file), output_dir, True)
                    for file in batch
                ]
                
                # Process results
                for future in futures:
                    try:
                        result = future.result()
                        if result is True:
                            processed_files += 1
                        else:
                            error_files += 1
                    except Exception as e:
                        error_files += 1
                        logging.error(f"Failed to process a file: {e}")
                
                # Ensure memory is freed between batches
                gc.collect()

    end_time = time.time()
    elapsed_time = end_time - start_time
    
    logging.info(f"Processing complete in {elapsed_time:.2f} seconds:")
    logging.info(f"Total files: {total_files}")
    logging.info(f"Already processed (skipped): {skipped_files}")
    logging.info(f"Successfully processed: {processed_files}")
    logging.info(f"Failed to process: {error_files}")
    
    if processed_files > 0:
        logging.info(f"Average processing time per file: {elapsed_time/processed_files:.2f} seconds")