import os
import sys
import re
import glob
import argparse
import json
from collections import defaultdict
import matplotlib.pyplot as plt

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
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), './origin_opensora')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), './')))

from opensora.acceleration.parallel_states import get_data_parallel_group
from opensora.datasets.dataloader import prepare_dataloader
from opensora.acceleration.parallel_states import set_sequence_parallel_group
from opensora.datasets.aspect import get_image_size, get_num_frames
from opensora.registry import MODELS, SCHEDULERS, DATASETS, build_module
from opensora.utils.config_utils import parse_configs
from opensora.utils.misc import create_logger, is_distributed, is_main_process, to_torch_dtype

def load_checkpoint(model, checkpoint_path, device):
    """Load checkpoint (ema.pt) for a model."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # Handle both regular checkpoint and EMA checkpoint
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)
    return model

def extract_checkpoint_info(checkpoint_path):
    """Extract epoch and step information from checkpoint path."""
    match = re.search(r'epoch(\d+)-global_step(\d+)', checkpoint_path)
    if match:
        epoch = int(match.group(1))
        step = int(match.group(2))
        return epoch, step
    return None, None

def calculate_metrics(ground_truth, prediction):
    """Calculate multiple image quality metrics."""
    # Ensure tensors are on CPU and in the right format
    if isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.detach().cpu()
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.detach().cpu()
    
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

def evaluate_checkpoint(cfg, checkpoint_path, validation_dataloader, device, dtype, logger):
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
    if is_distributed():
        coordinator = DistCoordinator()
        enable_sequence_parallelism = coordinator.world_size > 1
        if enable_sequence_parallelism:
            set_sequence_parallel_group(dist.group.WORLD)
    
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
    model = load_checkpoint(model, checkpoint_path, device)
    text_encoder.y_embedder = model.y_embedder
    
    # Build scheduler
    scheduler = build_module(cfg.scheduler, SCHEDULERS)
    
    # Evaluation
    all_metrics = defaultdict(list)
    batch_size = cfg.get("batch_size", 1)
    total_samples = 0
    
    # Extract checkpoint information for logging
    epoch, step = extract_checkpoint_info(checkpoint_path)
    
    with torch.no_grad():
        for batch in tqdm(validation_dataloader, desc=f"Evaluating epoch {epoch}, step {step}"):
            # Extract ground truth and brightness data
            ground_truth = batch["video"].to(device, dtype)
            brightness_data = batch["text"]
            
            # Generate samples from brightness data
            encoded = vae.encode(ground_truth)
            
            # Add noise (using the same approach as in training)
            torch.manual_seed(0)  # For reproducibility
            z = torch.randn_like(encoded)
            
            # Sample using scheduler
            samples = scheduler.sample(
                model,
                text_encoder,
                z=z,
                prompts=brightness_data,
                device=device,
                progress=False,
            )
            
            # Decode samples to pixel space
            generated_videos = vae.decode(samples.to(dtype), num_frames=num_frames)
            
            # Calculate metrics between ground truth and generated videos
            for i in range(len(ground_truth)):
                metrics = calculate_metrics(ground_truth[i], generated_videos[i])
                for metric_name, value in metrics.items():
                    all_metrics[metric_name].append(value)
            
            total_samples += len(ground_truth)
    
    # Calculate average metrics
    avg_metrics = {metric: sum(values) / len(values) for metric, values in all_metrics.items()}
    
    # Add checkpoint info
    avg_metrics["epoch"] = epoch
    avg_metrics["step"] = step
    
    logger.info(f"Finished evaluation of checkpoint (Epoch {epoch}, Step {step})")
    logger.info(f"Metrics: MAE={avg_metrics['mae']:.4f}, MSE={avg_metrics['mse']:.4f}, "
                f"PSNR={avg_metrics['psnr']:.2f}, SSIM={avg_metrics['ssim']:.4f}")
    
    return avg_metrics

def find_checkpoints(checkpoints_dir):
    """Find all checkpoint files (ema.pt) in the given directory and its subdirectories."""
    checkpoint_pattern = os.path.join(checkpoints_dir, "epoch*-global_step*", "ema.pt")
    return sorted(glob.glob(checkpoint_pattern))

def plot_metrics(metrics_list, save_path):
    """Plot metrics over training steps."""
    # Extract data
    epochs = [m['epoch'] for m in metrics_list]
    steps = [m['step'] for m in metrics_list]
    mae = [m['mae'] for m in metrics_list]
    mse = [m['mse'] for m in metrics_list]
    psnr = [m['psnr'] for m in metrics_list]
    ssim = [m['ssim'] for m in metrics_list]
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # MAE
    axes[0, 0].plot(steps, mae, 'b-o')
    axes[0, 0].set_title('Mean Absolute Error (MAE)')
    axes[0, 0].set_xlabel('Training Step')
    axes[0, 0].set_ylabel('MAE')
    axes[0, 0].grid(True)
    
    # MSE
    axes[0, 1].plot(steps, mse, 'r-o')
    axes[0, 1].set_title('Mean Squared Error (MSE)')
    axes[0, 1].set_xlabel('Training Step')
    axes[0, 1].set_ylabel('MSE')
    axes[0, 1].grid(True)
    
    # PSNR
    axes[1, 0].plot(steps, psnr, 'g-o')
    axes[1, 0].set_title('Peak Signal-to-Noise Ratio (PSNR)')
    axes[1, 0].set_xlabel('Training Step')
    axes[1, 0].set_ylabel('PSNR (dB)')
    axes[1, 0].grid(True)
    
    # SSIM
    axes[1, 1].plot(steps, ssim, 'm-o')
    axes[1, 1].set_title('Structural Similarity Index (SSIM)')
    axes[1, 1].set_xlabel('Training Step')
    axes[1, 1].set_ylabel('SSIM')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Evaluate T2V model performance across checkpoints')
    parser.add_argument('--config', type=str, default='Fine_tune/configs/train/fine_tune_stage1.py',
                        help='Path to the model configuration file')
    parser.add_argument('--checkpoints_dir', type=str, default='outputs/0002-Sunspot_STDiT3-XL-2',
                        help='Directory containing checkpoint directories')
    parser.add_argument('--validation_data_dir', type=str, default='/content/dataset/validation',
                        help='Directory containing validation data')
    parser.add_argument('--results_dir', type=str, default='evaluation_results',
                        help='Directory to save evaluation results')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for validation')
    parser.add_argument('--specific_checkpoint', type=str, default=None,
                        help='Evaluate only this specific checkpoint path')
    args = parser.parse_args()
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Set up environment
    torch.set_grad_enabled(False)
    
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
    
    # Device and dtype
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg_dtype = cfg.get("dtype", "fp32")
    dtype = to_torch_dtype(cfg.get("dtype", "bf16"))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Initialize distributed environment if necessary
    if is_distributed():
        colossalai.launch_from_torch({})
    set_random_seed(seed=cfg.get("seed", 1024))
    
    # Initialize logger
    logger = create_logger()
    logger.info("Evaluation configuration loaded")
    
    # Build dataset
    logger.info("Building validation dataset...")
    dataset = build_module(cfg.dataset, DATASETS)
    logger.info(f"Validation dataset contains {len(dataset)} samples.")
    
    # Build dataloader
    dataloader_args = dict(
        dataset=dataset,
        batch_size=cfg.get("batch_size", 1),
        num_workers=cfg.get("num_workers", 4),
        seed=cfg.get("seed", 1024),
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        process_group=get_data_parallel_group(),
        prefetch_factor=cfg.get("prefetch_factor", None),
    )
    validation_dataloader, _ = prepare_dataloader(
        bucket_config=cfg.get("bucket_config", None),
        num_bucket_build_workers=cfg.get("num_bucket_build_workers", 1),
        **dataloader_args,
    )
    
    # Find checkpoints or use specific checkpoint if provided
    if args.specific_checkpoint:
        checkpoints = [args.specific_checkpoint]
    else:
        checkpoints = find_checkpoints(args.checkpoints_dir)
    
    if not checkpoints:
        logger.error(f"No checkpoints found in {args.checkpoints_dir}")
        return
    
    logger.info(f"Found {len(checkpoints)} checkpoints to evaluate")
    
    # Evaluate each checkpoint
    all_metrics = []
    for checkpoint_path in checkpoints:
        metrics = evaluate_checkpoint(
            cfg, 
            checkpoint_path, 
            validation_dataloader, 
            device, 
            dtype, 
            logger
        )
        all_metrics.append(metrics)
    
    # Save results
    results_path = os.path.join(args.results_dir, 'metrics.json')
    with open(results_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    # Plot metrics
    plot_path = os.path.join(args.results_dir, 'metrics_plot.png')
    plot_metrics(all_metrics, plot_path)
    
    logger.info(f"Evaluation complete. Results saved to {args.results_dir}")
    
    # Print summary
    logger.info("Summary of results:")
    for metrics in all_metrics:
        logger.info(f"Epoch {metrics['epoch']}, Step {metrics['step']}: "
                    f"MAE={metrics['mae']:.4f}, MSE={metrics['mse']:.4f}, "
                    f"PSNR={metrics['psnr']:.2f}, SSIM={metrics['ssim']:.4f}")

if __name__ == "__main__":
    main()
