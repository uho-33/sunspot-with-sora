import os
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from tqdm import tqdm

def construct_brightness_dataset(input_file, output_dir, window_size, stride, start_date=None):
    """
    Construct a dataset from brightness time series data using sliding windows.
    
    Args:
        input_file (str): Path to the input NPZ file
        output_dir (str): Path to the output directory 
        window_size (int): Size of sliding window
        stride (int): Stride for sliding window
        start_date (str): Optional start date filter in format 'YYYY-MM-DD HH:MM:SS'
    """
    # Load the source data
    print(f"Loading data from {input_file}")
    data = np.load(input_file)
    
    # Extract time series data and timestamps
    brightness_data = data['normalized_brightness']
    timestamps = data['timestamps']
    
    # Print basic information
    print(f"Loaded data shape: {brightness_data.shape}")
    print(f"Timestamp count: {len(timestamps)}")
    print(f"First timestamp: {timestamps[0]}")
    print(f"Last timestamp: {timestamps[-1]}")
    
    # Filter data by start date if specified
    if start_date:
        start_datetime = datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S")
        print(f"Filtering data after {start_date}")
        
        # Convert timestamps to datetime objects for comparison
        filtered_indices = []
        for i, ts in enumerate(timestamps):
            # Handle different timestamp formats
            if isinstance(ts, (datetime, np.datetime64)):
                ts_datetime = pd.to_datetime(ts)
            else:
                try:
                    ts_datetime = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
                except Exception:
                    print(f"Warning: Could not parse timestamp {ts}, skipping")
                    continue
                
            if ts_datetime >= start_datetime:
                filtered_indices.append(i)
        
        if not filtered_indices:
            raise ValueError(f"No data found after {start_date}")
            
        # Filter data and timestamps
        brightness_data = brightness_data[filtered_indices]
        timestamps = timestamps[filtered_indices]
        
        print(f"Filtered data shape: {brightness_data.shape}")
        print(f"Filtered timestamp count: {len(timestamps)}")
        if len(timestamps) > 0:
            print(f"First filtered timestamp: {timestamps[0]}")
            print(f"Last filtered timestamp: {timestamps[-1]}")
    
    # Create output directory
    dataset_dir = Path(output_dir)
    os.makedirs(dataset_dir, exist_ok=True)
    print(f"Output directory: {dataset_dir}")
    
    # Apply sliding window to create multiple NPZ files
    num_windows = max(1, (len(brightness_data) - window_size) // stride + 1)
    print(f"Creating {num_windows} window samples...")
    
    for i in tqdm(range(num_windows)):
        # Calculate start and end indices for this window
        start_idx = i * stride
        end_idx = min(start_idx + window_size, len(brightness_data))
        
        # Skip if window is smaller than window_size
        if end_idx - start_idx < window_size:
            continue
            
        # Get window data
        window_data = brightness_data[start_idx:end_idx]
        window_timestamps = timestamps[start_idx:end_idx]
        
        # Format the timestamp for the filename
        # Assuming timestamps are in string format like "YYYY-MM-DD HH:MM:SS"
        try:
            timestamp_str = window_timestamps[0]
            # If timestamp is a datetime object
            if isinstance(timestamp_str, (datetime, np.datetime64)):
                timestamp_str = pd.to_datetime(timestamp_str).strftime("%Y%m%d_%H%M%S")
            # If timestamp is already a string but in a different format
            else:
                dt = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                timestamp_str = dt.strftime("%Y%m%d_%H%M%S")
        except Exception as e:
            print(f"Error formatting timestamp {window_timestamps[0]}: {e}")
            # Use index as filename if timestamp conversion fails
            timestamp_str = f"window_{i:06d}"
        
        # Create output filename
        output_file = dataset_dir / f"{timestamp_str}.npz"
        
        # Save window data
        np.savez(
            output_file,
            data=window_data,
            timestamps=window_timestamps,
            window_idx=i,
            window_size=window_size,
            stride=stride
        )
    
    print(f"Dataset creation complete. {num_windows} window samples saved to {dataset_dir}")
    return dataset_dir

def main():
    parser = argparse.ArgumentParser(description='Construct brightness dataset with sliding windows')
    parser.add_argument('--input_file', 
                        default='data/processed/brightness/Ic_720s_normalize_dn_brightness_timeseries.npz',
                        help='Path to input NPZ file containing brightness time series')
    parser.add_argument('--target_dir', 
                        default=None,
                        help='Directory to save the dataset. If not provided, will use "dataset/training/brightness/L{window_size}-S{stride}"')
    parser.add_argument('--window_size', type=int, default=16,
                        help='Size of sliding window (number of frames)')
    parser.add_argument('--stride', type=int, default=8,
                        help='Stride for sliding window')
    parser.add_argument('--start_date', type=str, default="2021-11-01 00:00:00",
                        help='Only include data after this date (format: YYYY-MM-DD HH:MM:SS)')
    
    args = parser.parse_args()
    
    # Convert relative paths to absolute paths based on project root
    input_file = args.input_file
    if args.target_dir is None:
        date_str = args.start_date.split()[0].replace('-', '')
        output_dir = f'dataset/training/brightness/L{args.window_size}-S{args.stride}'
    else:
        output_dir = args.target_dir + f'/L{args.window_size}-S{args.stride}'
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Construct dataset
    try:
        dataset_dir = construct_brightness_dataset(
            input_file=input_file,
            output_dir=output_dir,
            window_size=args.window_size,
            stride=args.stride,
            start_date=args.start_date
        )
        print(f"Successfully created dataset at {dataset_dir}")
    except Exception as e:
        print(f"Error constructing dataset: {e}")
        raise

if __name__ == '__main__':
    import pandas as pd  # Import pandas for timestamp handling
    main()
