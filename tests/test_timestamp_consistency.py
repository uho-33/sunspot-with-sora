import os
import re
import numpy as np
import pandas as pd
from datetime import datetime
import pytest
def test_timestamps_consistency():
    """t to verify that every image in the 360p directory has a corresponding
    Test to verify that every image in the 360p directory has a corresponding
    timestamp in the brightness timeseries data.
    """
    # Paths
    image_dir = r"data\processed\figure\figure-downsample\360p_filtered"
    brightness_file = r"data\processed\brightness\filtered\filtered_brightness_timeseries.npz"
    # Get all image filenames
    # Get all image filenames
    image_files = os.listdir(image_dir)
    image_files = [f for f in image_files if f.endswith('.png') and 'continuum_nd' in f]
    # Extract timestamps from filenames
    # Extract timestamps from filenames
    timestamp_pattern = re.compile(r'\.(\d{8}_\d{6})_TAI\.')
    image_timestamps = []
    
    for filename in image_files:
        match = timestamp_pattern.search(filename)
        if match:
            timestamp_str = match.group(1)
            # Convert to datetime objecttamp_str, "%Y%m%d_%H%M%S")
            dt = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
            image_timestamps.append(dt)
    # Load brightness timeseries data
    # Load brightness timeseries data
    brightness_data = np.load(brightness_file)
    brightness_timestamps = brightness_data['timestamps']
    
    # Handle conversion of timestamps correctly based on their typee9) for ts in brightness_timestamps]
    brightness_datetimes = []
    if np.issubdtype(brightness_timestamps.dtype, np.datetime64):
        # If they are np.datetime64, convert to Python datetime
        brightness_datetimes = [pd.Timestamp(ts).to_pydatetime() for ts in brightness_timestamps]
    else:
        # If they are numeric timestamps (seconds or nanoseconds since epoch)    matches = [abs((img_dt - b_dt).total_seconds()) < 1 for b_dt in brightness_datetimes]
        # Convert to nanoseconds first if needed
        if np.max(brightness_timestamps) > 1e18:  # Already in nanosecondsmps.append(img_dt)
            brightness_datetimes = [datetime.fromtimestamp(ts / 1e9) for ts in brightness_timestamps]
        else:  # In seconds
            brightness_datetimes = [datetime.fromtimestamp(ts) for ts in brightness_timestamps]
    
    # Check if each image timestamp has a corresponding brightness timestamp
    missing_timestamps = []
    for img_dt in image_timestamps:
        # Find if there's a matching timestamp within a small tolerance (e.g., 1 second)
        matches = [abs((img_dt - b_dt).total_seconds()) < 1 for b_dt in brightness_datetimes]    # Check if differences are consistent (within a small tolerance)
        if not any(matches):
            missing_timestamps.append(img_dt)

    # Assert no missing timestampsuniform. Mean: {mean_diff} hours, StdDev: {std_diff}"
    assert len(missing_timestamps) == 0, f"Missing timestamps for {len(missing_timestamps)} images"
    
    # Verify time unit consistency
    if brightness_timestamps.size > 1:
    # Calculate time differences between consecutive timestamps (in hours)        if np.issubdtype(brightness_timestamps.dtype, np.datetime64):
        # For datetime64 arrays, convert differences to timedeltas in hours
        time_diffs = np.diff(brightness_timestamps).astype('timedelta64[s]').astype(float) / 3600
    else:
        # For numeric timestamps, calculate differences in hours
        if np.max(brightness_timestamps) > 1e18:  # nanoseconds
            time_diffs = np.diff(brightness_timestamps) / (1e9 * 3600)
        else:  # seconds
            time_diffs = np.diff(brightness_timestamps) / 3600
    
        # Check if differences are consistent (within a small tolerance)
        mean_diff = np.mean(time_diffs)
        std_diff = np.std(time_diffs)
        # A small standard deviation relative to the mean indicates consistent time spacing
        assert std_diff / mean_diff < 0.01, f"Time intervals are not uniform. Mean: {mean_diff} hours, StdDev: {std_diff}"
        print(f"Time differences are consistent with mean interval of {mean_diff:.2f} hours")

if __name__ == "__main__":
    test_timestamps_consistency()
