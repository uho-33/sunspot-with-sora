import os
import re
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import shutil

def filter_matching_data():
    """
    Filter data to keep only entries that have corresponding files in both the image directory
    and the brightness timeseries data. Also standardizes timestamps to yyyymmdd_hhmmss format.
    """
    # Paths
    image_dir = r"data\processed\figure\figure-downsample\360p"
    brightness_file = r"data\processed\brightness\Ic_720s_normalize_dn_brightness_timeseries.npz"
    output_dir = brightness_file + "/../" +"filtered"
    filtered_image_dir = image_dir + "/../" + "360p_filtered"
    
    # Ensure output directories exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(filtered_image_dir, exist_ok=True)
    
    # Get all image filenames
    image_files = os.listdir(image_dir)
    image_files = [f for f in image_files if f.endswith('.png') and 'continuum_nd' in f]
    
    # Extract timestamps from filenames
    timestamp_pattern = re.compile(r'\.(\d{8}_\d{6})_TAI\.')
    image_timestamps = {}
    
    for filename in image_files:
        match = timestamp_pattern.search(filename)
        if match:
            timestamp_str = match.group(1)
            # Convert to datetime object
            dt = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
            # Store with original filename
            image_timestamps[dt] = filename
    
    # Load brightness timeseries data
    brightness_data = np.load(brightness_file)
    brightness_timestamps = brightness_data['timestamps']
    brightness_values = brightness_data['normalized_brightness']
    
    # Convert brightness timestamps to datetime objects
    brightness_datetimes = []
    if np.issubdtype(brightness_timestamps.dtype, np.datetime64):
        # If they are np.datetime64, convert to Python datetime
        brightness_datetimes = [pd.Timestamp(ts).to_pydatetime() for ts in brightness_timestamps]
    else:
        # If they are numeric timestamps (seconds or nanoseconds since epoch)
        if np.max(brightness_timestamps) > 1e18:  # Already in nanoseconds
            brightness_datetimes = [datetime.fromtimestamp(ts / 1e9) for ts in brightness_timestamps]
        else:  # In seconds
            brightness_datetimes = [datetime.fromtimestamp(ts) for ts in brightness_timestamps]
    
    # Find matching timestamps
    matched_data = []
    matched_images = []
    
    print(f"Total image files: {len(image_timestamps)}")
    print(f"Total brightness timestamps: {len(brightness_datetimes)}")
    
    for idx, b_dt in enumerate(brightness_datetimes):
        # Standardize timestamp to yyyymmdd_hhmmss format, truncating smaller values
        standardized_dt = datetime(b_dt.year, b_dt.month, b_dt.day, b_dt.hour, b_dt.minute, b_dt.second)
        
        # Find if there's a matching image timestamp within a small tolerance (e.g., 1 second)
        for img_dt, img_filename in image_timestamps.items():
            if abs((img_dt - standardized_dt).total_seconds()) < 1:
                # Format timestamp as yyyymmdd_hhmmss
                formatted_ts = standardized_dt.strftime("%Y%m%d_%H%M%S")
                
                # Save the matching data
                matched_data.append((formatted_ts, brightness_values[idx]))
                matched_images.append((img_filename, formatted_ts))
                break
    
    print(f"Found {len(matched_data)} matches")
    
    if matched_data:
        # Save filtered brightness data
        timestamps = np.array([item[0] for item in matched_data])
        brightness = np.array([item[1] for item in matched_data])
        
        filtered_brightness_file = os.path.join(output_dir, "filtered_brightness_timeseries.npz")
        np.savez(filtered_brightness_file, timestamps=timestamps, brightness=brightness)
        print(f"Saved filtered brightness data to {filtered_brightness_file}")
        
        # Copy matching images to filtered directory
        for img_filename, formatted_ts in matched_images:
            src_path = os.path.join(image_dir, img_filename)
            # Rename file to use standardized timestamp
            new_filename = img_filename.replace(
                re.search(timestamp_pattern, img_filename).group(1), 
                formatted_ts
            )
            dst_path = os.path.join(filtered_image_dir, new_filename)
            shutil.copy2(src_path, dst_path)
        
        print(f"Copied {len(matched_images)} matching images to {filtered_image_dir}")
        
        # Calculate and save statistics for the filtered data
        mean = np.mean(brightness)
        std = np.std(brightness)
        stats_file = os.path.join(output_dir, "filtered_brightness_stats.npz")
        np.savez(stats_file, mean=mean, std=std)
        print(f"Saved mean ({mean:.2e}) and std ({std:.2e}) to {stats_file}")

if __name__ == "__main__":
    filter_matching_data()
