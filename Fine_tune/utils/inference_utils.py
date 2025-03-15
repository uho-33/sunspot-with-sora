import os
import torch
import torchvision.utils as vutils
from pathlib import Path

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