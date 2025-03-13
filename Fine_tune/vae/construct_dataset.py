import os
import re
import shutil
from pathlib import Path
import argparse
from datetime import datetime
from tqdm import tqdm

def extract_datetime(filename):
    """Extract datetime from filename like '360p_hmi.ic_nolimbdark_720s.20211101_000000_TAI.3.continuum_nd.png'"""
    match = re.search(r'(\d{8}_\d{6})_TAI', filename)
    if match:
        date_str = match.group(1)
        return date_str
    return None

def construct_time_series_dataset(source_dir, target_dir, window_size=16, stride=8):
    """
    Construct dataset with sliding window approach
    
    Args:
        source_dir: Directory containing source images
        target_dir: Directory to save the dataset
        window_size: Number of consecutive frames in each data point
        stride: Step size for sliding window
    """
    # Create target directory if it doesn't exist
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    
    # Get all PNG files from source directory
    source_path = Path(source_dir)
    image_files = sorted([f for f in source_path.glob("*.png")])
    
    print(f"Found {len(image_files)} images in {source_dir}")
    
    # Extract timestamps and sort files chronologically
    timestamp_files = []
    for img_path in image_files:
        timestamp = extract_datetime(img_path.name)
        if timestamp:
            timestamp_files.append((timestamp, img_path))
    
    # Sort by timestamp
    timestamp_files.sort(key=lambda x: x[0])
    
    # Check if we have enough files
    if len(timestamp_files) < window_size:
        print(f"Not enough images for a single window (found {len(timestamp_files)}, need {window_size})")
        return
    
    # Create data points with sliding window
    data_point_count = 0
    for i in range(0, len(timestamp_files) - window_size + 1, stride):
        # Get window of files
        window = timestamp_files[i:i+window_size]
        
        # Get timestamp of first image for folder name
        first_timestamp = window[0][0]
        
        # Create folder named after timestamp
        data_point_dir = target_path / first_timestamp
        data_point_dir.mkdir(exist_ok=True)
        
        # Copy images to the folder
        for j, (_, img_path) in enumerate(window):
            # Use a numerical prefix to maintain order
            output_name = f"{j:03d}_{img_path.name}"
            shutil.copy2(img_path, data_point_dir / output_name)
        
        data_point_count += 1
        
    print(f"Created {data_point_count} data points in {target_dir}")

def main():
    parser = argparse.ArgumentParser(description='Construct time series dataset for sun images')
    parser.add_argument('--source_dir', type=str, default='data/processed/figure/figure-downsample/360p_filtered',
                        help='Directory containing source images')
    parser.add_argument('--target_dir', type=str, default='dataset/training/figure',
                        help='Directory to save the dataset. If not provided, will use "dataset/training/figure/L{window_size}-S{stride}"')
    parser.add_argument('--window_size', type=int, default=16,
                        help='Number of consecutive frames in each data point')
    parser.add_argument('--stride', type=int, default=8,
                        help='Step size for sliding window')
    
    args = parser.parse_args()
    
    # Determine target directory based on window size and stride if not provided
    if args.target_dir is None:
        raise ValueError("--target_dir can not be none")
    else:
        output_dir = args.target_dir + f"/L{args.window_size}-S{args.stride}"
    
    print(f"Constructing dataset with window size {args.window_size} and stride {args.stride}")
    print(f"Output directory: {output_dir}")
    construct_time_series_dataset(args.source_dir, output_dir, args.window_size, args.stride)
    print("Dataset construction completed!")

if __name__ == "__main__":
    main()
