import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

def load_data(data_path, stats_path):
    """Load time series data and statistics."""
    print(f"Loading time series from: {data_path}")
    print(f"Loading statistics from: {stats_path}")
    
    # Load time series data
    timeseries_data = np.load(data_path)
    timestamps = timeseries_data['timestamps']
    brightness = timeseries_data['brightness']
    
    # Load statistics
    stats_data = np.load(stats_path)
    mean = stats_data['mean']
    std = stats_data['std']
    
    print(f"Loaded {len(timestamps)} data points")
    print(f"Statistics: mean={mean:.2e}, std={std:.2e}")
    
    return timestamps, brightness, mean, std

def normalize_brightness(brightness, mean, std):
    """Normalize brightness values using z-score normalization."""
    return (brightness - mean) / std

def main():
    """Main function to normalize brightness time series."""
    os.chdir(r"E:\study-and-research\sunspot-with-sora")
    
    # File paths
    save_path = "data/processed/"
    prefix = "Ic_720s_normalize_distance"
    
    timeseries_path = save_path + prefix + "total_brightness_timeseries.npz"
    stats_path = save_path + prefix + "brightness_stats.npz"
    
    # Load data
    timestamps, brightness, mean, std = load_data(timeseries_path, stats_path)
    
    # Normalize brightness
    normalized_brightness = normalize_brightness(brightness, mean, std)
    
    # Save normalized data
    normalized_path = save_path + prefix + "normalized_brightness_timeseries.npz"
    np.savez(normalized_path,
             timestamps=timestamps,
             brightness=brightness,
             normalized_brightness=normalized_brightness,
             mean=mean,
             std=std)
    
    print(f"Saved normalized data to {normalized_path}")
    
    # Create and save visualization
    plt.figure(figsize=(14, 8))
    
    # Plot both original and normalized brightness
    plt.subplot(2, 1, 1)
    plt.plot(timestamps, brightness)
    plt.title('Original Brightness Time Series')
    plt.ylabel('Brightness')
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(timestamps, normalized_brightness)
    plt.title('Normalized Brightness Time Series')
    plt.xlabel('Time')
    plt.ylabel('Z-Score')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path + prefix + "brightness_comparison.png")
    print(f"Saved visualization to {save_path + prefix + 'brightness_comparison.png'}")
    plt.show()

if __name__ == "__main__":
    main()
