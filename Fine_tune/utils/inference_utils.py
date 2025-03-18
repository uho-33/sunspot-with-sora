import os
import torch
import torchvision.utils as vutils
from pathlib import Path
import pandas as pd
import numpy as np

def modified_get_save_path_name(
    save_dir,
    sample_name=None,  # prefix
    sample_idx=None,  # sample index
    prompt=None,  # used prompt
    prompt_filename=None,
    prompt_as_path=False,  # use prompt as path
    prompt_filename_as_path=False,
    num_sample=1,  # number of samples to generate for one prompt
    k=None,  # kth sample
):
    if sample_name is None:
        sample_name = "" if prompt_as_path else "sample"
    if prompt_as_path:
        sample_name_suffix = prompt
    elif prompt_filename_as_path:
        sample_name_suffix = prompt_filename
    else:
        sample_name_suffix = f"_{sample_idx:04d}"
    save_path = os.path.join(save_dir, f"{sample_name}{sample_name_suffix}")
    if num_sample != 1:
        save_path = f"{save_path}-{k}"
    return save_path



def save_video_as_frames(video_tensor, save_dir, prefix="frame_", verbose=True ):
    """
    Save a video tensor [C, T, H, W] as individual image frames.
    
    Args:
        video_tensor (torch.Tensor): Video tensor of shape [C, T, H, W]
        save_dir (str): Directory to save the frames
        prefix (str): Prefix for filenames (default: "frame_")
        verbose(bool):  print information
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Get dimensions
    C, T, H, W = video_tensor.shape
    print(f"Saving {T} frames to {save_dir}")
    
    # Save each frame as a separate image
    for t in range(T):
        frame = video_tensor[:, t, :, :]  # Extract frame at position t [C, H, W]
        frame_path = os.path.join(save_dir, f"{prefix}{t:04d}.png")
        
        # Use torchvision.utils.save_image with correct parameters
        vutils.save_image(
            frame, 
            frame_path,
            normalize=True,
            value_range=(-1, 1)
        )
    if verbose:
        print(f"Saved to {save_dir}")
    return save_dir



def load_brightness_prompts(prompt_path, start_idx=None, end_idx=None, csv_ref_column_name=None, first_column_as_name=False):
    """
    Load brightness prompts from various file formats.
    Returns:
        prompts: List where each element is a data series (list or array of values)
        names: List of names corresponding to each data series
    """
    prompts = []
    names = []

    # Handle directory: process all files in directory
    if os.path.isdir(prompt_path):
        all_results = []
        all_names = []
        for file in sorted(os.listdir(prompt_path)):
            file_path = os.path.join(prompt_path, file)
            if os.path.isfile(file_path) and (file_path.endswith('.npz') or file_path.endswith('.csv')):
                file_prompts, file_names = load_brightness_prompts(
                    file_path, start_idx=None, end_idx=None, 
                    csv_ref_column_name=csv_ref_column_name,
                    first_column_as_name=first_column_as_name
                )
                all_results.extend(file_prompts)
                all_names.extend(file_names)
        
        # Apply start and end indices only at the directory level
        if start_idx is not None or end_idx is not None:
            all_results = all_results[start_idx:end_idx]
            all_names = all_names[start_idx:end_idx]
        
        return all_results, all_names

    # Handle .npz file
    elif prompt_path.endswith(".npz"):
        data = np.load(prompt_path)
        if 'data' in data:
            # Ensure data is 2D: [num_series, values_per_series]
            loaded_data = data['data']
            if loaded_data.ndim == 1:
                # If 1D, treat as a single series
                prompts = [loaded_data.tolist()]
            else:
                # If already 2D or higher, use as is
                prompts = [series.tolist() for series in loaded_data]
                
            # Use basename without extension as name
            base_name = os.path.basename(prompt_path).split('.')[0]
            names = [f"{base_name}_{i}" for i in range(len(prompts))]
        else:
            raise ValueError(f"NPZ file {prompt_path} does not contain 'data' key")
    
    # Handle .csv file
    elif prompt_path.endswith(".csv"):
        df = pd.read_csv(prompt_path)
        
        if first_column_as_name:
            # Use first column as names and remaining columns as data
            name_col = df.columns[0]
            names = df[name_col].tolist()
            
            # Get all other columns as data
            if len(df.columns) > 1:
                data_cols = df.columns[1:]
                # Each row is a separate data series
                prompts = df[data_cols].values.tolist()
            else:
                # If only one column, use empty prompts
                prompts = [[] for _ in range(len(names))]
        else:
            # Use indices as names
            if "text" in df.columns:
                # For compatibility with text prompts
                text_data = df["text"].tolist()
                # Convert each text entry to a list of numbers if possible
                prompts = []
                for text in text_data:
                    try:
                        # Try parsing as list of numbers
                        if isinstance(text, str) and '[' in text and ']' in text:
                            import ast
                            values = ast.literal_eval(text)
                        else:
                            # For single values
                            values = [float(text)]
                        prompts.append(values)
                    except (ValueError, SyntaxError):
                        # If not numeric, keep as is for text-based generation
                        prompts.append([0.0])  # Default placeholder
            else:
                # If no "text" column, use all data
                # Each row becomes a data series
                prompts = df.values.tolist()
            
            base_name = os.path.basename(prompt_path).split('.')[0]
            names = [f"{base_name}_{i}" for i in range(len(prompts))]

        # Handle reference paths if specified
        if csv_ref_column_name is not None:
            assert csv_ref_column_name in df, f"column {csv_ref_column_name} for reference paths not found in {prompt_path}"
            reference_paths = df[csv_ref_column_name].tolist()
            return prompts, names, reference_paths
    else:
        raise ValueError(f"Unsupported file format for {prompt_path}. Supported formats: .npz, .csv")

    # Apply start and end indices
    if start_idx is not None or end_idx is not None:
        prompts = prompts[start_idx:end_idx]
        names = names[start_idx:end_idx]

    return prompts, names



def process_brightness_data(brightness_data, logger):
    """Process brightness data to ensure it's in the expected format."""
    try:
        if isinstance(brightness_data, torch.Tensor):
            # Convert brightness tensors to lists of numeric values
            if brightness_data.dim() > 1:
                # Convert each tensor row to a list of float values
                formatted_prompts = []
                for b in range(brightness_data.size(0)):
                    # Convert to list of float values (not strings)
                    values = brightness_data[b].cpu().tolist()
                    formatted_prompts.append(values)
                return formatted_prompts
            else:
                # For 1D tensor, convert to list directly
                return [brightness_data.cpu().tolist()]
        elif isinstance(brightness_data, list):
            # Check if the list contains tensors
            if brightness_data and isinstance(brightness_data[0], torch.Tensor):
                return [b.cpu().tolist() if isinstance(b, torch.Tensor) else b for b in brightness_data]
        # Return as is if it's already in a suitable format
        return brightness_data
    except Exception as e:
        logger.error(f"Error formatting brightness data: {e}")
        try:
            # Try converting to float lists
            return [[float(x) for x in b] for b in brightness_data]
        except:
            logger.error("Failed to convert to float lists, using numeric constants")
            # Last resort: use simple numeric values
            if isinstance(brightness_data, list):
                return [0.5] * len(brightness_data)  # Default value as fallback
            else:
                return [0.5]  # Single default value
