import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import sys
import traceback

def process_single_file(file_path):
    """Process a single file to debug transformation issues."""
    print(f"Processing file: {file_path}")
    
    # Load the data
    with h5py.File(file_path, 'r') as hf:
        data = hf['data'][:]
    
    # Calculate basic statistics
    data_min = np.nanmin(data)
    data_max = np.nanmax(data)
    data_mean = np.nanmean(data)
    data_std = np.nanstd(data)
    valid_count = np.count_nonzero(~np.isnan(data))
    nan_count = np.count_nonzero(np.isnan(data))
    
    print("\nOriginal Data Statistics:")
    print(f"Min: {data_min}")
    print(f"Max: {data_max}")
    print(f"Mean: {data_mean}")
    print(f"Std: {data_std}")
    print(f"Valid points: {valid_count} ({valid_count/data.size*100:.2f}%)")
    print(f"NaN points: {nan_count} ({nan_count/data.size*100:.2f}%)")
    
    # Load brightness stats (similar to figure-norm.py)
    try:
        # Try different possible paths for the brightness stats
        print("\nLooking for brightness stats file...")
        stats_path = None
        
        # Option 1
        path1 = Path("data/processed/Ic_720s_normalize_distancebrightness_stats.npz")
        if path1.exists():
            stats_path = path1
            print(f"Found stats file at: {stats_path}")
            
        # Option 2
        if not stats_path:
            for path in Path("data/processed").glob("*brightness_stats.npz"):
                stats_path = path
                print(f"Found stats file at: {stats_path}")
                break
                
        # Option 3 - direct path for testing
        if not stats_path:
            path3 = Path("data/processed/Ic_720s_normalize_distance/brightness_stats.npz")
            if path3.exists():
                stats_path = path3
                print(f"Found stats file at: {stats_path}")
        
        # Option 4 - other common locations
        if not stats_path:
            for path in Path("data/processed").rglob("*brightness_stats.npz"):
                stats_path = path
                print(f"Found stats file at: {stats_path}")
                break
        
        if not stats_path:
            print("Could not find brightness stats file. Creating dummy values for testing.")
            mean_brightness = 4.5e11  # Example value
            std_brightness = 1.0e10   # Example value
        else:
            print(f"Loading stats from: {stats_path}")
            stats = np.load(stats_path)
            
            # Print available keys in the npz file
            print(f"Available keys in stats file: {list(stats.keys())}")
            
            # Try to get the mean and std values
            if 'mean' in stats:
                mean_brightness = stats['mean']
                print(f"Mean brightness from file: {mean_brightness}")
            else:
                print("Warning: 'mean' not found in stats file, using default")
                mean_brightness = 4.5e11
                
            if 'std' in stats:
                std_brightness = stats['std']
                print(f"Std brightness from file: {std_brightness}")
            else:
                print("Warning: 'std' not found in stats file, using default")
                std_brightness = 1.0e10
        
        # Calculate transformation parameters
        n = valid_count  # For single image analysis, use its own count
        m_o = mean_brightness / n
        sigma_o = std_brightness / n
        sigma = sigma_o/m_o
        m=1
        
        print("\nCalculated transformation parameters:")
        print(f"n (valid points): {n}")
        print(f"mean_brightness: {mean_brightness}")
        print(f"std_brightness: {std_brightness}")
        print(f"m = mean_brightness/n: {m}")
        print(f"sigma = std_brightness/n: {sigma}")
        
        # Calculate the transformation bounds
        lower_bound = m - 2 * sigma
        upper_bound = m + 2 * sigma
        dynamic_range = 4 * sigma
        
        print(f"\nTransformation bounds:")
        print(f"Lower bound (m - 2*sigma): {lower_bound}")
        print(f"Upper bound (m + 2*sigma): {upper_bound}")
        print(f"Dynamic range (4*sigma): {dynamic_range}")
        
        # Check how data fits within these bounds
        below_lower = np.sum(data < lower_bound) / valid_count * 100
        above_upper = np.sum(data > upper_bound) / valid_count * 100
        
        print(f"\nData distribution relative to bounds:")
        print(f"Below lower bound: {below_lower:.2f}%")
        print(f"Above upper bound: {above_upper:.2f}%")
        print(f"Within bounds: {100 - below_lower - above_upper:.2f}%")
        
        # Create test transformations
        nan_replaced_data = np.nan_to_num(data, nan=0)
        
        print("\nCreating transformed images...")
        # Original transformation
        transformed_original = np.clip((nan_replaced_data - (m - 2 * sigma)) / (4 * sigma) * 255, 0, 255).astype(np.uint8)
        
        # Alternative transformations for comparison
        transformed_minmax = np.clip((nan_replaced_data - data_min) / (data_max - data_min) * 255, 0, 255).astype(np.uint8)
        transformed_meanstd = np.clip((nan_replaced_data - (data_mean - 2 * data_std)) / (4 * data_std) * 255, 0, 255).astype(np.uint8)
        
        # Check output statistics
        print("\nTransformed data statistics:")
        print(f"Original transformation - Min: {transformed_original.min()}, Max: {transformed_original.max()}, Mean: {transformed_original.mean():.2f}, Unique values: {len(np.unique(transformed_original))}")
        print(f"Min-max transformation - Min: {transformed_minmax.min()}, Max: {transformed_minmax.max()}, Mean: {transformed_minmax.mean():.2f}, Unique values: {len(np.unique(transformed_minmax))}")
        print(f"Mean-std transformation - Min: {transformed_meanstd.min()}, Max: {transformed_meanstd.max()}, Mean: {transformed_meanstd.mean():.2f}, Unique values: {len(np.unique(transformed_meanstd))}")
        
        # Visualize the results
        print("\nCreating visualization plots...")
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Original data
        im0 = axes[0, 0].imshow(data, cmap='viridis')
        axes[0, 0].set_title('Original Data')
        plt.colorbar(im0, ax=axes[0, 0])
        
        # Original transformation
        im1 = axes[0, 1].imshow(transformed_original, cmap='gray')
        axes[0, 1].set_title('Original Transformation')
        plt.colorbar(im1, ax=axes[0, 1])
        
        # Min-max transformation
        im2 = axes[0, 2].imshow(transformed_minmax, cmap='gray')
        axes[0, 2].set_title('Min-Max Transformation')
        plt.colorbar(im2, ax=axes[0, 2])
        
        # Mean-std transformation
        im3 = axes[1, 0].imshow(transformed_meanstd, cmap='gray')
        axes[1, 0].set_title('Mean-Std Transformation')
        plt.colorbar(im3, ax=axes[1, 0])
        
        # Histogram of original data
        axes[1, 1].hist(data[~np.isnan(data)].flatten(), bins=50)
        axes[1, 1].axvline(x=lower_bound, color='r', linestyle='--', label='Lower bound')
        axes[1, 1].axvline(x=upper_bound, color='g', linestyle='--', label='Upper bound')
        axes[1, 1].set_title('Histogram of Original Data')
        axes[1, 1].legend()
        
        # Histogram of transformed data
        axes[1, 2].hist(transformed_original.flatten(), bins=50, alpha=0.5, label='Original')
        axes[1, 2].hist(transformed_minmax.flatten(), bins=50, alpha=0.5, label='Min-Max')
        axes[1, 2].hist(transformed_meanstd.flatten(), bins=50, alpha=0.5, label='Mean-Std')
        axes[1, 2].set_title('Histogram of Transformed Data')
        axes[1, 2].legend()
        
        plt.tight_layout()
        
        # Save the results
        output_dir = Path("data/processed/figure/debug")
        output_dir.mkdir(parents=True, exist_ok=True)
        base_name = Path(file_path).stem
        
        print(f"\nSaving output to {output_dir}")
        plt.savefig(output_dir / f"{base_name}_analysis.png")
        
        # Save each transformed image
        Image.fromarray(transformed_original).save(output_dir / f"{base_name}_original_transform.png")
        Image.fromarray(transformed_minmax).save(output_dir / f"{base_name}_minmax_transform.png")
        Image.fromarray(transformed_meanstd).save(output_dir / f"{base_name}_meanstd_transform.png")
        
        print(f"\nResults saved to {output_dir}")
        return True
        
    except Exception as e:
        print(f"Error during transformation analysis: {str(e)}")
        print("\nDetailed traceback:")
        traceback.print_exc()
        return False

def main():
    print("Starting script...")
    try:
        os.chdir(r"E:\study-and-research\sunspot-with-sora")
        print(f"Working directory: {os.getcwd()}")
        
        # Get the first h5 file or use command line argument
        if len(sys.argv) > 1:
            file_path = sys.argv[1]
        else:
            input_dir = Path("data/processed/Ic_nolimbdark_720s_normalize_distance") 
            print(f"Looking for h5 files in: {input_dir}")
            h5_files = list(input_dir.glob("*.h5"))
            if not h5_files:
                print("No h5 files found!")
                return
            file_path = str(h5_files[0])
            print(f"No file specified, using first file: {file_path}")
        
        success = process_single_file(file_path)
        
        if success:
            print("\nDisplaying plots (close window to exit)...")
            plt.show()
        
    except Exception as e:
        print(f"Error in main function: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
