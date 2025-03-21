import os
import sys
import re
import glob
import argparse
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import logging
import datetime
from typing import Dict, List, Tuple, Any
import pickle
import traceback

# Set environment variables to disable bitsandbytes before importing other libraries
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
os.environ["BNB_DISABLE_QUANTIZATION"] = "1"

import colossalai
import torch
import torch.distributed as dist
import torch.nn.functional as F
from colossalai.cluster import DistCoordinator
from mmengine.runner import set_random_seed
from tqdm import tqdm
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# Add paths to pythonpath
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../origin_opensora')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from opensora.acceleration.parallel_states import get_data_parallel_group
from opensora.datasets.dataloader import prepare_dataloader
from opensora.acceleration.parallel_states import set_sequence_parallel_group
from opensora.datasets.aspect import get_image_size, get_num_frames
from opensora.registry import MODELS, SCHEDULERS, DATASETS, build_module
from opensora.utils.config_utils import parse_configs
from opensora.utils.misc import to_torch_dtype

# Custom JSON encoder to handle NumPy data types
class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, 
                           np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.void):
            return None
        return super(NumpyEncoder, self).default(obj)

# Create a basic logger that doesn't depend on distributed status
def setup_basic_logger():
    logger = logging.getLogger("checkpoint_eval")
    logger.setLevel(logging.INFO)
    
    # Create console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Create formatter - Fix the typo in the formatter pattern
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(ch)
    
    return logger

def extract_checkpoint_info(checkpoint_path):
    """Extract epoch and step information from checkpoint path."""
    # Try the original format first (epoch-global_step)
    match = re.search(r'epoch(\d+)-global_step(\d+)', checkpoint_path)
    if match:
        epoch = int(match.group(1))
        step = int(match.group(2))
        return epoch, step
    
    # Try the new format (step2000.pt)
    match = re.search(r'step(\d+)\.pt', os.path.basename(checkpoint_path))
    if match:
        step = int(match.group(1))
        # Since there's no epoch information, set epoch to 0
        return 0, step
    
    return None, None

def calculate_metrics(ground_truth, prediction):
    """Calculate multiple image quality metrics."""
    # Ensure tensors are on CPU and convert to float32
    if isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.detach().cpu().to(torch.float)
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.detach().cpu().to(torch.float)
    
    # Handle frame count mismatch
    if ground_truth.shape[1] != prediction.shape[1]:
        # Get minimum number of frames from both videos
        min_frames = min(ground_truth.shape[1], prediction.shape[1])
        # Use only the first min_frames for both videos
        ground_truth = ground_truth[:, :min_frames]
        prediction = prediction[:, :min_frames]
    
    # Calculate Mean Absolute Error (MAE)
    mae = F.l1_loss(prediction, ground_truth).item()
    
    # Calculate Mean Squared Error (MSE)
    mse = F.mse_loss(prediction, ground_truth).item()
    
    # Convert to numpy for skimage metrics
    gt_np = ground_truth.numpy()
    pred_np = prediction.numpy()
    
    # Calculate PSNR - handle frame by frame
    psnr_values = []
    ssim_values = []
    
    # For video data [C, T, H, W], calculate metrics per frame
    for t in range(gt_np.shape[1]):
        # Get individual frames - [C, H, W]
        gt_frame = gt_np[:, t]
        pred_frame = pred_np[:, t]
        
        # Convert to format expected by skimage metrics
        # Normalize to [0, 1] range if needed
        gt_frame = np.transpose(gt_frame, (1, 2, 0))  # [H, W, C]
        pred_frame = np.transpose(pred_frame, (1, 2, 0))  # [H, W, C]
        
        # Calculate PSNR
        psnr_val = psnr(gt_frame, pred_frame, data_range=1.0)
        psnr_values.append(psnr_val)
        
        # Calculate SSIM
        ssim_val = ssim(gt_frame, pred_frame, data_range=1.0, channel_axis=2, multichannel=True)
        ssim_values.append(ssim_val)
    
    return {
        'mae': mae,
        'mse': mse,
        'psnr': np.mean(psnr_values),
        'ssim': np.mean(ssim_values)
    }

def evaluate_checkpoint(cfg, checkpoint_path, validation_dataloader, device, dtype, logger, is_main=True, results_dir=None):
    """Evaluate a single checkpoint on the validation data."""
    logger.info(f"Evaluating checkpoint: {checkpoint_path}")
    
    # Build models
    text_encoder = build_module(cfg.text_encoder, MODELS, device=device)
    vae = build_module(cfg.vae, MODELS).to(device, dtype).eval()
    
    # Prepare video size
    image_size = cfg.get("image_size", (240, 240))
    num_frames = get_num_frames(cfg.get("num_frames", 16))
    
    # Build diffusion model
    input_size = (num_frames, *image_size)
    latent_size = vae.get_latent_size(input_size)
    enable_sequence_parallelism = False
    if dist.is_initialized():
        coordinator = DistCoordinator()
        enable_sequence_parallelism = coordinator.world_size > 1
        if enable_sequence_parallelism:
            set_sequence_parallel_group(dist.group.WORLD)
    
    cfg.model["from_pretrained"] = checkpoint_path
    print(f"from_pretrained={cfg.model['from_pretrained']}")
    model = build_module(
        cfg.model,
        MODELS,
        input_size=latent_size,
        in_channels=vae.out_channels,
        caption_channels=text_encoder.output_dim,
        model_max_length=text_encoder.model_max_length,
        enable_sequence_parallelism=enable_sequence_parallelism,
    ).to(device, dtype).eval()
    
    # Load checkpoint
    text_encoder.y_embedder = model.y_embedder
    
    # Patch for y_embedder null method in fourier text encoder
    if hasattr(text_encoder, 'null'):
        original_null = text_encoder.null
        # Create a patched null method
        def patched_null(n):
            try:
                # Try the original method first
                return original_null(n)
            except RuntimeError as e:
                logger.warning(f"Original null method failed: {e}")
                # Debug y_embedder structure
                y_embedding = text_encoder.y_embedder.y_embedding
                logger.info(f"y_embedding shape: {y_embedding.shape}, dtype: {y_embedding.dtype}")
                
                # Ensure we return a 4D tensor with shape [n, 1, seq_len, embed_dim]
                if y_embedding.dim() == 1:
                    # For 1D embeddings (embed_dim,), expand to 4D
                    embed_dim = y_embedding.shape[0]
                    null_y = y_embedding.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(n, 1, 1, 1)
                    logger.info(f"Created null_y with shape: {null_y.shape} from 1D embedding")
                    return null_y
                elif y_embedding.dim() == 2:
                    # For 2D embeddings [seq_len, embed_dim], add batch and channel dimensions
                    seq_len, embed_dim = y_embedding.shape
                    # Convert to [n, 1, seq_len, embed_dim]
                    null_y = y_embedding.unsqueeze(0).unsqueeze(1).repeat(n, 1, 1, 1)
                    logger.info(f"Created null_y with shape: {null_y.shape} from 2D embedding")
                    return null_y
                else:
                    # For other cases, create a zero tensor with expected shape
                    logger.warning(f"Creating fallback zero tensor for null embedding")
                    # Try to infer the expected shape from encoder output_dim
                    embed_dim = getattr(text_encoder, "output_dim", 4096)
                    seq_len = getattr(text_encoder, "model_max_length", 300)
                    # Create 4D tensor [n, 1, seq_len, embed_dim]
                    return torch.zeros(n, 1, seq_len, embed_dim, device=device, dtype=dtype)
                
        # Replace the null method
        text_encoder.null = patched_null
    
    # Build scheduler
    scheduler = build_module(cfg.scheduler, SCHEDULERS)
    
    # Evaluation
    all_metrics = defaultdict(list)
    per_sample_metrics = []  # Store metrics for each sample with timestamps
    batch_size = cfg.get("batch_size", 1)
    total_samples = 0
    sample_idx = 0  # Track overall sample index
    
    # Extract checkpoint information for logging
    epoch, step = extract_checkpoint_info(checkpoint_path)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(validation_dataloader, desc=f"Evaluating epoch {epoch}, step {step}")):
            # Extract ground_truth and brightness data
            ground_truth = batch["video"].to(device, dtype)
            brightness_data = batch["text"]
            
            # Generate samples from brightness data
            encoded = vae.encode(ground_truth)
            
            # Add noise (using the same approach as in training)
            torch.manual_seed(0)  # For reproducibility
            z = torch.randn_like(encoded)
            
            # Process brightness data to ensure it's in the expected format for the text encoder
            if isinstance(brightness_data, torch.Tensor):
                # Convert brightness tensors to lists of numeric values
                try:
                    # Check if we're dealing with numeric brightness values
                    if brightness_data.dim() > 1:
                        # Convert each tensor row to a list of float values
                        formatted_prompts = []
                        for b in range(brightness_data.size(0)):
                            # Convert to list of float values (not strings)
                            values = brightness_data[b].cpu().tolist()
                            formatted_prompts.append(values)  # Pass as a list, not a string
                        brightness_data = formatted_prompts
                    else:
                        # For 1D tensor, convert to list directly
                        brightness_data = [brightness_data.cpu().tolist()]
                except Exception as e:
                    logger.error(f"Error formatting brightness data: {e}")
                    # Fall back to simple float representation if the above fails
                    try:
                        brightness_data = [[float(x) for x in b] for b in brightness_data]
                    except:
                        logger.error("Failed to convert to float lists, using numeric constants")
                        # Last resort: use simple numeric values
                        brightness_data = [0.5] * len(brightness_data)  # Default value as fallback
            elif isinstance(brightness_data, list):
                # Check if the list contains tensors
                if brightness_data and isinstance(brightness_data[0], torch.Tensor):
                    brightness_data = [b.cpu().tolist() if isinstance(b, torch.Tensor) else b for b in brightness_data]
            
            try:
                # Get tokenized data for the text encoder
                tokens = text_encoder.tokenize_fn(brightness_data)
                encoded_data = text_encoder.encode(**tokens) if tokens else None
                
                # Extract video dimensions from the batch
                batch_size = ground_truth.shape[0]
                height = batch["height"].to(device) if "height" in batch else torch.ones(batch_size, device=device) * ground_truth.shape[2]
                width = batch["width"].to(device) if "width" in batch else torch.ones(batch_size, device=device) * ground_truth.shape[3]
                num_frames = batch["num_frames"].to(device) if "num_frames" in batch else torch.ones(batch_size, device=device) * ground_truth.shape[1]
                
                # Create model_kwargs dictionary required by scheduler
                model_kwargs = {
                    "height": height,
                    "width": width,
                    "num_frames": num_frames,
                    "fps": batch["fps"].to(device) if "fps" in batch else torch.ones(batch_size, device=device) * 16.0,  # Add fps parameter
                }
                
                # Sample using scheduler with properly formatted brightness data
                samples = scheduler.sample(
                    model,
                    text_encoder,
                    z=z,
                    prompts=brightness_data,
                    device=device,
                    progress=False,
                    additional_args=model_kwargs,  # This parameter name changed from model_kwargs to additional_args
                )
                
                # Decode samples to pixel space
                generated_videos = vae.decode(samples.to(dtype), num_frames=num_frames[0].item())
                
                # Calculate metrics between ground truth and generated videos
                for i in range(len(ground_truth)):
                    metrics = calculate_metrics(ground_truth[i], generated_videos[i])
                    
                    # Store per-sample metrics
                    per_sample_metrics.append({
                        'sample_idx': sample_idx,
                        'metrics': metrics
                    })
                    
                    # Add aggregate metrics as well
                    for metric_name, value in metrics.items():
                        all_metrics[metric_name].append(value)
                    
                    # Increment sample index after processing each sample
                    sample_idx += 1
                
                total_samples += len(ground_truth)
                
            except Exception as e:
                logger.error(f"Error during sampling: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                logger.error("Skipping this batch due to error")
                continue  # Skip to the next batch
    
    # Calculate average metrics
    if all_metrics:
        avg_metrics = {metric: sum(values) / len(values) for metric, values in all_metrics.items()}
        
        # Add checkpoint info
        avg_metrics["epoch"] = epoch
        avg_metrics["step"] = step
        
        logger.info(f"Finished evaluation of checkpoint (Epoch {epoch}, Step {step})")
        logger.info(f"Metrics: MAE={avg_metrics['mae']:.4f}, MSE={avg_metrics['mse']:.4f}, "
                    f"PSNR={avg_metrics['psnr']:.2f}, SSIM={avg_metrics['ssim']:.4f}")
    else:
        # If no metrics were collected, return a default structure with just epoch and step
        logger.warning("No metrics were collected for this checkpoint (all batches failed)")
        avg_metrics = {
            "epoch": epoch,
            "step": step,
            "mae": 0.0,
            "mse": 0.0,
            "psnr": 0.0,
            "ssim": 0.0,
            "error": "All batches failed"
        }
    
    # Return both averaged metrics and per-sample metrics
    return {
        "avg_metrics": avg_metrics,
        "per_sample_metrics": per_sample_metrics
    }

def find_checkpoints(checkpoints_dir, checkpoint_name="ema.pt", step_interval=None):
    """
    Find all checkpoint files in the given directory.
    
    Args:
        checkpoints_dir: Directory containing checkpoint files
        checkpoint_name: Name pattern of checkpoint files (ignored if files are like step*.pt)
        step_interval: If provided, sample checkpoints at this step interval
    
    Returns:
        List of checkpoint paths
    """
    # Check if we're using the new format in heatmap_ckp directory
    if os.path.basename(checkpoints_dir) == "heatmap_ckp" or "heatmap_ckp" in checkpoints_dir:
        # Look for files with pattern "step*.pt"
        checkpoint_pattern = os.path.join(checkpoints_dir, "step*.pt")
        all_checkpoints = glob.glob(checkpoint_pattern)
    else:
        # Original format
        checkpoint_pattern = os.path.join(checkpoints_dir, "epoch*-global_step*", checkpoint_name)
        all_checkpoints = glob.glob(checkpoint_pattern)
    
    if not all_checkpoints:
        print(f"Warning: No checkpoints found with pattern {checkpoint_pattern}")
    
    if not step_interval or step_interval <= 0:
        # Sort checkpoints by their step numbers
        return sorted(all_checkpoints, key=lambda x: extract_checkpoint_info(x)[1] or 0)
    
    # Extract step info for each checkpoint and sort by step
    ckpt_with_info = []
    for ckpt in all_checkpoints:
        epoch, step = extract_checkpoint_info(ckpt)
        if step is not None:  # Make sure we have at least the step info
            ckpt_with_info.append((ckpt, epoch, step))
    
    # Sort by step number
    ckpt_with_info.sort(key=lambda x: x[2])
    
    if not ckpt_with_info:
        return []
    
    # Always include the first and last checkpoint
    sampled_checkpoints = [ckpt_with_info[0][0]]
    
    # Sample intermediate checkpoints at the specified interval
    start_step = ckpt_with_info[0][2]
    last_included_step = start_step
    for ckpt, epoch, step in ckpt_with_info[1:-1]:
        if step - last_included_step >= step_interval:
            sampled_checkpoints.append(ckpt)
            last_included_step = step
    
    # Always include the last checkpoint if we have more than one
    if len(ckpt_with_info) > 1:
        sampled_checkpoints.append(ckpt_with_info[-1][0])
    
    return sampled_checkpoints

def create_metrics_heatmap(all_checkpoint_metrics, metric_name, save_path):
    """
    Create a 2D heatmap visualization of metrics for each sample across different checkpoints.
    
    Args:
        all_checkpoint_metrics: List of metrics dictionaries for each checkpoint
        metric_name: Name of the metric to visualize (mae, mse, psnr, ssim)
        save_path: Path to save the heatmap image
    """
    # Extract checkpoint steps for y-axis
    checkpoint_steps = [metrics_dict["avg_metrics"]["step"] for metrics_dict in all_checkpoint_metrics]
    
    # Find the maximum number of samples in any checkpoint
    max_samples = max(len(metrics_dict["per_sample_metrics"]) for metrics_dict in all_checkpoint_metrics)
    
    # Create a matrix to hold the metric values
    # Shape: [num_checkpoints, max_samples]
    heatmap_data = np.zeros((len(all_checkpoint_metrics), max_samples))
    
    # Fill the matrix with NaN values initially
    heatmap_data[:] = np.nan
    
    # Fill the matrix with metric values for each checkpoint and sample
    for i, metrics_dict in enumerate(all_checkpoint_metrics):
        for sample in metrics_dict["per_sample_metrics"]:
            sample_idx = sample["sample_idx"]
            if sample_idx < max_samples:
                value = sample["metrics"][metric_name]
                heatmap_data[i, sample_idx] = value
    
    # Create the heatmap visualization
    plt.figure(figsize=(max(12, max_samples // 10), 8))
    
    # Choose colormap based on whether higher or lower values are better
    cmap = 'viridis'
    if metric_name in ['mae', 'mse']:  # Lower is better
        cmap = 'viridis_r'
    
    # Create the heatmap
    im = plt.imshow(heatmap_data, aspect='auto', cmap=cmap)
    
    # Add colorbar
    plt.colorbar(im, label=f'{metric_name.upper()} value')
    
    # Set labels and title
    plt.xlabel('Sample Index (Sequential)')
    plt.ylabel('Checkpoint Step')
    plt.title(f'{metric_name.upper()} per Sample Across Checkpoints')
    
    # Set ticks for y-axis (checkpoint steps)
    plt.yticks(np.arange(len(checkpoint_steps)), 
               [str(step) for step in checkpoint_steps])
    
    # Limit the number of x-ticks to avoid overcrowding
    x_tick_step = max(1, max_samples // 20)
    plt.xticks(np.arange(0, max_samples, x_tick_step))
    
    # Add grid
    plt.grid(False)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    return heatmap_data

def save_checkpoint_result(results_dir, checkpoint_metrics, checkpoint_path, logger):
    """Save results for a single checkpoint evaluation."""
    # Create a unique filename for this checkpoint
    _, step = extract_checkpoint_info(checkpoint_path)
    result_filename = f"checkpoint_step_{step}_results.json"
    result_path = os.path.join(results_dir, result_filename)
    
    try:
        # Save the checkpoint metrics
        with open(result_path, 'w') as f:
            json.dump(checkpoint_metrics, f, indent=2, cls=NumpyEncoder)
        logger.info(f"Saved checkpoint results to {result_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save checkpoint results: {e}")
        return False

def load_previous_results(results_dir, logger):
    """Load all previously saved checkpoint results."""
    checkpoint_results = []
    
    if not os.path.exists(results_dir):
        return checkpoint_results
    
    result_files = glob.glob(os.path.join(results_dir, "checkpoint_step_*_results.json"))
    
    if not result_files:
        return checkpoint_results
    
    logger.info(f"Found {len(result_files)} previous checkpoint results")
    
    for result_file in result_files:
        try:
            with open(result_file, 'r') as f:
                checkpoint_result = json.load(f)
                checkpoint_results.append(checkpoint_result)
            logger.info(f"Loaded checkpoint result from {result_file}")
        except Exception as e:
            logger.error(f"Failed to load result from {result_file}: {e}")
    
    return checkpoint_results

def get_already_evaluated_steps(results_dir):
    """Get list of step numbers already evaluated."""
    evaluated_steps = []
    
    if not os.path.exists(results_dir):
        return evaluated_steps
    
    result_files = glob.glob(os.path.join(results_dir, "checkpoint_step_*_results.json"))
    
    for result_file in result_files:
        try:
            # Extract step number from filename
            step_match = re.search(r'checkpoint_step_(\d+)_results.json', os.path.basename(result_file))
            if step_match:
                step = int(step_match.group(1))
                evaluated_steps.append(step)
        except Exception:
            pass
    
    return evaluated_steps

def update_visualizations(all_metrics, all_checkpoint_metrics, results_dir, use_wandb, logger):
    """Update visualization files with current results."""
    if not all_metrics:
        logger.warning("No metrics to visualize")
        return
    
    logger.info("Updating visualization files...")
    
    # Create directory for heatmaps
    heatmap_dir = os.path.join(results_dir, 'heatmaps')
    os.makedirs(heatmap_dir, exist_ok=True)
    
    # Create heatmaps for each metric
    metrics_to_plot = ['mae', 'mse', 'psnr', 'ssim']
    for metric_name in metrics_to_plot:
        heatmap_path = os.path.join(heatmap_dir, f'{metric_name}_heatmap.png')
        logger.info(f"Creating {metric_name} heatmap...")
        
        # Create and save the heatmap
        heatmap_data = create_metrics_heatmap(
            all_checkpoint_metrics,
            metric_name,
            heatmap_path
        )
        logger.info(f"Saved {metric_name} heatmap to {heatmap_path}")
        
        # Log to wandb if enabled
        if use_wandb:
            try:
                import wandb
                wandb.log({
                    f"heatmaps/{metric_name}": wandb.Image(heatmap_path),
                    "global_step": max([m["avg_metrics"]["step"] for m in all_checkpoint_metrics])
                })
            except Exception as e:
                logger.warning(f"Failed to log heatmap to wandb: {e}")
    
    # Create traditional line plots to show progression of average metrics
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot([m["step"] for m in all_metrics], [m["mae"] for m in all_metrics], 'o-')
    plt.title('Mean Absolute Error (MAE)')
    plt.xlabel('Training Step')
    plt.ylabel('MAE')
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot([m["step"] for m in all_metrics], [m["mse"] for m in all_metrics], 'o-')
    plt.title('Mean Squared Error (MSE)')
    plt.xlabel('Training Step')
    plt.ylabel('MSE')
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.plot([m["step"] for m in all_metrics], [m["psnr"] for m in all_metrics], 'o-')
    plt.title('Peak Signal-to-Noise Ratio (PSNR)')
    plt.xlabel('Training Step')
    plt.ylabel('PSNR (dB)')
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.plot([m["step"] for m in all_metrics], [m["ssim"] for m in all_metrics], 'o-')
    plt.title('Structural Similarity Index (SSIM)')
    plt.xlabel('Training Step')
    plt.ylabel('SSIM')
    plt.grid(True)
    
    plt.tight_layout()
    line_plot_path = os.path.join(results_dir, 'metrics_progression.png')
    plt.savefig(line_plot_path)
    plt.close()
    
    if use_wandb:
        try:
            import wandb
            wandb.log({
                "metrics_progression": wandb.Image(line_plot_path),
                "global_step": max([m["step"] for m in all_metrics])
            })
        except Exception as e:
            logger.warning(f"Failed to log metrics progression to wandb: {e}")

def main():
    # First, set up a basic logger that doesn't depend on distributed status
    logger = setup_basic_logger()
    
    parser = argparse.ArgumentParser(description='Evaluate T2V model performance across checkpoints')
    parser.add_argument('--config', type=str, default='Fine_tune/configs/train/evaluate.py',
                        help='Path to the model configuration file')
    parser.add_argument('--checkpoints_dir', type=str, default='outputs/heatmap_ckp',
                        help='Directory containing checkpoint files')
    parser.add_argument('--validation_data_dir', type=str, default='/content/dataset/validation',
                        help='Directory containing validation data')
    parser.add_argument('--results_dir', type=str, default='outputs/evaluation/time_dependency',
                        help='Directory to save evaluation results')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for validation')
    parser.add_argument('--specific_checkpoint', type=str, default=None,
                        help='Evaluate only this specific checkpoint path')
    parser.add_argument('--use_wandb', action='store_true',
                        help='Enable logging to Weights & Biases')
    parser.add_argument('--wandb_project', type=str, default='sun-reconstruction-eval',
                        help='Weights & Biases project name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='Weights & Biases entity/team name')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                        help='Weights & Biases run name')
    parser.add_argument('--checkpoint_name', type=str, default="ema.pt",
                        help='used checkpoint type, default ema.pt (ignored for heatmap_ckp)')
    parser.add_argument('--step_interval', type=int, default=0,
                        help='Interval between steps to sample checkpoints (0 = use all checkpoints)')
    parser.add_argument('--disable_bnb', action='store_true', default=True,
                        help='Disable bitsandbytes quantization for compatibility with older systems')
    args = parser.parse_args()
    
    # Disable bitsandbytes if requested
    if args.disable_bnb:
        logger.info("Disabling bitsandbytes quantization")
        os.environ["BITSANDBYTES_NOWELCOME"] = "1"
        os.environ["BNB_DISABLE_QUANTIZATION"] = "1"
    
    # Log initial information including all argument values for debugging
    logger.info("Starting evaluation script")
    logger.info(f"Arguments: config={args.config}, checkpoints_dir={args.checkpoints_dir}")
    logger.info(f"Results will be saved to: {args.results_dir}")
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Set up environment
    torch.set_grad_enabled(False)
    
    # Handle bitsandbytes import issues
    try:
        # Try to disable torch cuda memory efficient attention if it's causing issues
        if args.disable_bnb:
            import transformers
            if hasattr(transformers, "utils") and hasattr(transformers.utils, "fx"):
                logger.info("Disabling torch cuda memory efficient attention")
                transformers.utils.fx._DISABLE_TORCH_CUDA_MEMORY_EFFICIENT_ATTENTION = True
    except:
        logger.warning("Could not disable transformers memory efficient attention")
    
    # Parse configs
    sys.argv = ['', args.config]
    cfg = parse_configs(training=False)
    
    # Override configuration settings for validation
    cfg.dataset = dict(
        type="SunObservationDataset",
        transform_name="center",
        time_series_dir=os.path.join(args.validation_data_dir, "figure/360p/L16-S8/"),
        brightness_dir=os.path.join(args.validation_data_dir, "brightness/L16-S8/"),
    )
    cfg.batch_size = args.batch_size
    
    # Update model settings to avoid quantization
    if args.disable_bnb and hasattr(cfg, 'model'):
        if isinstance(cfg.model, dict):
            cfg.model["load_in_8bit"] = False
            cfg.model["load_in_4bit"] = False
            if "dtype" in cfg.model:
                logger.info(f"Setting model dtype to {cfg.dtype} instead of quantized")
                cfg.model["dtype"] = cfg.dtype
    
    # Device and dtype
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = to_torch_dtype(cfg.get("dtype", "bf16"))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Initialize distributed environment
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_distributed_run = False
    is_main = True  # Default to True for non-distributed case
    
    logger.info(f"Environment setup: LOCAL_RANK={local_rank}, WORLD_SIZE={world_size}")
    
    try:
        if world_size > 1:
            # Initialize PyTorch distributed directly
            torch.cuda.set_device(local_rank)
            logger.info(f"Initializing process group with {world_size} processes")
            dist.init_process_group(backend="nccl")
            is_distributed_run = dist.is_initialized()
            is_main = dist.get_rank() == 0
            logger.info(f"Process group initialized: rank={dist.get_rank()}, world_size={dist.get_world_size()}")
            
            # Initialize colossalai if needed
            if is_distributed_run:
                try:
                    logger.info("Initializing ColossalAI from torch")
                    colossalai.launch_from_torch({})
                    logger.info("ColossalAI initialized successfully")
                except Exception as e:
                    logger.warning(f"Could not initialize colossalai: {e}. Continuing without it.")
    except Exception as e:
        logger.error(f"Failed to initialize distributed environment: {e}")
        # Continue with non-distributed mode
        is_distributed_run = False
        is_main = True
    
    logger.info(f"Distributed setup complete: is_distributed={is_distributed_run}, is_main={is_main}")
    
    set_random_seed(seed=cfg.get("seed", 1024))
    
    # Initialize wandb for basic metrics tracking if requested
    if is_main and args.use_wandb:
        try:
            import wandb
            model_name = os.path.basename(args.checkpoints_dir)
            wandb_run_name = args.wandb_run_name or f"evaluation_{model_name}"
            
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=wandb_run_name,
                config={
                    "model_dir": args.checkpoints_dir,
                    "validation_data_dir": args.validation_data_dir,
                    "batch_size": args.batch_size,
                    "config_file": args.config,
                }
            )
            logger.info(f"Wandb initialized with run: {wandb.run.name}")
        except Exception as e:
            logger.error(f"Error initializing wandb: {e}")
            args.use_wandb = False
    
    logger.info("Evaluation configuration loaded")
    
    try:
        # Build dataset
        logger.info("Building validation dataset...")
        dataset = build_module(cfg.dataset, DATASETS)
        logger.info(f"Validation dataset contains {len(dataset)} samples.")
        
        # Build dataloader
        logger.info("Creating validation dataloader...")
        
        # Base dataloader arguments
        dataloader_args = dict(
            dataset=dataset,
            batch_size=cfg.get("batch_size", 1),
            num_workers=cfg.get("num_workers", 4),
            seed=cfg.get("seed", 1024),
            shuffle=False,  # Important: keep samples in sequential order
            drop_last=False,
            pin_memory=True,
            prefetch_factor=cfg.get("prefetch_factor", None),
        )
        
        # Handle bucket config args
        bucket_args = {}
        if cfg.get("bucket_config") is not None:
            bucket_args["bucket_config"] = cfg.get("bucket_config")
            bucket_args["num_bucket_build_workers"] = cfg.get("num_bucket_build_workers", 1)
        
        if is_distributed_run:
            # Get data parallel group for distributed training
            logger.info("Using distributed dataloader setup")
            process_group = get_data_parallel_group()
            dataloader_args["process_group"] = process_group
            
            validation_dataloader, _ = prepare_dataloader(
                **dataloader_args,
                **bucket_args
            )
        else:
            # Create a simple PyTorch dataloader for non-distributed case
            logger.info("Using non-distributed dataloader setup")
            from torch.utils.data import DataLoader
            
            # Try using OpenSora's dataloader without process_group
            try:
                logger.info("Attempting to use OpenSora dataloader without process group")
                validation_dataloader, _ = prepare_dataloader(
                    **dataloader_args,
                    **bucket_args
                )
            except (ValueError, RuntimeError) as e:
                logger.warning(f"Could not use OpenSora dataloader: {e}")
                logger.info("Falling back to standard PyTorch DataLoader")
                validation_dataloader = DataLoader(
                    dataset,
                    batch_size=dataloader_args["batch_size"],
                    num_workers=dataloader_args["num_workers"],
                    shuffle=dataloader_args["shuffle"],
                    drop_last=dataloader_args["drop_last"],
                    pin_memory=dataloader_args["pin_memory"],
                )
        
        logger.info("Dataloader created successfully")
        
        # Find checkpoints or use specific checkpoint if provided
        if args.specific_checkpoint:
            checkpoints = [args.specific_checkpoint]
        else:
            logger.info(f"Finding checkpoints in {args.checkpoints_dir}")
            checkpoints = find_checkpoints(
                args.checkpoints_dir, 
                args.checkpoint_name,
                args.step_interval
            )
            # Sort checkpoints by step number
            checkpoints.sort(key=lambda x: extract_checkpoint_info(x)[1] or 0)
        
        if not checkpoints:
            logger.error(f"No checkpoints found in {args.checkpoints_dir}")
            return
        
        logger.info(f"Found {len(checkpoints)} checkpoints to evaluate")
        for i, ckpt in enumerate(checkpoints):
            epoch, step = extract_checkpoint_info(ckpt)
            logger.info(f"Checkpoint {i+1}/{len(checkpoints)}: Epoch {epoch}, Step {step}")
        
        # Load previously evaluated checkpoints and results
        logger.info("Checking for previous evaluation results...")
        all_checkpoint_metrics = load_previous_results(args.results_dir, logger)
        evaluated_steps = get_already_evaluated_steps(args.results_dir)
        all_metrics = [result["avg_metrics"] for result in all_checkpoint_metrics]
        
        logger.info(f"Found {len(all_checkpoint_metrics)} previously evaluated checkpoints")
        
        # Determine which checkpoints need evaluation
        to_evaluate = []
        for ckpt in checkpoints:
            _, step = extract_checkpoint_info(ckpt)
            if step not in evaluated_steps:
                to_evaluate.append(ckpt)
        
        logger.info(f"Need to evaluate {len(to_evaluate)} new checkpoints")
        
        # Evaluate each checkpoint that hasn't been processed yet
        for checkpoint_path in to_evaluate:
            metrics_result = evaluate_checkpoint(
                cfg, 
                checkpoint_path, 
                validation_dataloader, 
                device, 
                dtype, 
                logger,
                is_main=is_main,
                results_dir=args.results_dir
            )
            
            # Store both average and per-sample metrics
            all_metrics.append(metrics_result["avg_metrics"])
            all_checkpoint_metrics.append(metrics_result)
            
            # Save this checkpoint's results immediately
            save_checkpoint_result(args.results_dir, metrics_result, checkpoint_path, logger)
            
            # Save the aggregate results so far
            if is_main:
                # Save average metrics
                avg_results_path = os.path.join(args.results_dir, 'metrics.json')
                with open(avg_results_path, 'w') as f:
                    json.dump(all_metrics, f, indent=2, cls=NumpyEncoder)
                
                # Save detailed metrics
                detailed_results_path = os.path.join(args.results_dir, 'detailed_metrics.json')
                with open(detailed_results_path, 'w') as f:
                    json.dump(all_checkpoint_metrics, f, indent=2, cls=NumpyEncoder)
                
                # Update visualizations every time we evaluate a checkpoint
                update_visualizations(all_metrics, all_checkpoint_metrics, args.results_dir, args.use_wandb, logger)
            
            # Log to wandb if enabled
            if is_main and args.use_wandb:
                epoch, step = extract_checkpoint_info(checkpoint_path)
                avg_metrics = metrics_result["avg_metrics"]
                
                # Add a global key that tracks the step number being evaluated
                wandb_metrics = {
                    f"metrics/mae": avg_metrics["mae"],
                    f"metrics/mse": avg_metrics["mse"],
                    f"metrics/psnr": avg_metrics["psnr"],
                    f"metrics/ssim": avg_metrics["ssim"],
                    f"checkpoint/step": step,
                    f"checkpoint/epoch": epoch,
                    "global_step": step  # Add this for better step tracking in wandb
                }
                
                try:
                    import wandb
                    wandb.log(wandb_metrics)
                except Exception as e:
                    logger.error(f"Failed to log metrics to wandb: {e}")
        
        # Now that all checkpoints are evaluated, generate the final visualizations
        if is_main and all_metrics:
            update_visualizations(all_metrics, all_checkpoint_metrics, args.results_dir, args.use_wandb, logger)
            
            # Print summary
            logger.info("Summary of results:")
            for metrics in all_metrics:
                logger.info(f"Epoch {metrics['epoch']}, Step {metrics['step']}: "
                            f"MAE={metrics['mae']:.4f}, MSE={metrics['mse']:.4f}, "
                            f"PSNR={metrics['psnr']:.2f}, SSIM={metrics['ssim']:.4f}")
            
            # Find best checkpoint for each metric
            best_mae_idx = min(range(len(all_metrics)), key=lambda i: all_metrics[i]["mae"])
            best_mse_idx = min(range(len(all_metrics)), key=lambda i: all_metrics[i]["mse"])
            best_psnr_idx = max(range(len(all_metrics)), key=lambda i: all_metrics[i]["psnr"]) 
            best_ssim_idx = max(range(len(all_metrics)), key=lambda i: all_metrics[i]["ssim"])
            
            logger.info(f"Best MAE: {all_metrics[best_mae_idx]['mae']:.4f} (Step {all_metrics[best_mae_idx]['step']})")
            logger.info(f"Best MSE: {all_metrics[best_mse_idx]['mse']:.4f} (Step {all_metrics[best_mse_idx]['step']})")
            logger.info(f"Best PSNR: {all_metrics[best_psnr_idx]['psnr']:.2f} (Step {all_metrics[best_psnr_idx]['step']})")
            logger.info(f"Best SSIM: {all_metrics[best_ssim_idx]['ssim']:.4f} (Step {all_metrics[best_ssim_idx]['step']})")
            
            # Log best metrics to wandb
            if args.use_wandb:
                try:
                    import wandb
                    best_metrics = {
                        "best/mae": all_metrics[best_mae_idx]["mae"],
                        "best/mse": all_metrics[best_mse_idx]["mse"],
                        "best/psnr": all_metrics[best_psnr_idx]["psnr"],
                        "best/ssim": all_metrics[best_ssim_idx]["ssim"],
                        "best_step/mae": all_metrics[best_mae_idx]["step"],
                        "best_step/mse": all_metrics[best_mse_idx]["step"],
                        "best_step/psnr": all_metrics[best_psnr_idx]["step"],
                        "best_step/ssim": all_metrics[best_ssim_idx]["step"],
                    }
                    wandb.log(best_metrics)
                    
                    # Set the best metrics as summary metrics
                    for key, value in best_metrics.items():
                        if not key.startswith("best_step"):
                            wandb.run.summary[key] = value
                except Exception as e:
                    logger.error(f"Failed to log best metrics to wandb: {e}")
    
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        logger.error(traceback.format_exc())
    
    finally:
        # Cleanup
        if is_distributed_run:
            logger.info("Cleaning up distributed environment")
            dist.destroy_process_group()
        
        # Finish wandb if initialized
        if is_main and args.use_wandb:
            logger.info("Finishing wandb session")
            # Log a summary of the best metrics
            if all_metrics:
                best_metrics = {
                    "best/mae": all_metrics[best_mae_idx]["mae"],
                    "best/mse": all_metrics[best_mse_idx]["mse"],
                    "best/psnr": all_metrics[best_psnr_idx]["psnr"],
                    "best/ssim": all_metrics[best_ssim_idx]["ssim"],
                    "best_step/mae": all_metrics[best_mae_idx]["step"],
                    "best_step/mse": all_metrics[best_mse_idx]["step"],
                    "best_step/psnr": all_metrics[best_psnr_idx]["step"],
                    "best_step/ssim": all_metrics[best_ssim_idx]["step"],
                }
                wandb.log(best_metrics)
                
                # Set the best metrics as summary metrics (shown at the top of the run page)
                for key, value in best_metrics.items():
                    if not key.startswith("best_step"):
                        wandb.run.summary[key] = value
            
            wandb.finish()
        
        logger.info("Script completed")

if __name__ == "__main__":
    main()
