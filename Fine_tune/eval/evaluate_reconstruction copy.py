import os
import glob
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
import colossalai
import torch.distributed as dist
from colossalai.cluster import DistCoordinator

from opensora.registry import MODELS, DATASETS, build_module
from opensora.utils.config_utils import parse_configs
from opensora.utils.misc import create_logger, is_distributed, to_torch_dtype
from opensora.datasets.dataloader import prepare_dataloader


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate reconstruction loss across checkpoints")
    parser.add_argument("--config", type=str, required=True, help="Path to the model config file")
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Directory containing checkpoint folders")
    parser.add_argument("--val_dir", type=str, default="/content/dataset/validation", help="Validation data directory")
    parser.add_argument("--output", type=str, default="reconstruction_results.csv", help="Results output file path")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run evaluation on")
    return parser.parse_args()


def find_checkpoints(ckpt_dir):
    """Find all checkpoint folders containing ema.pt files"""
    results = []
    for dirpath, _, filenames in os.walk(ckpt_dir):
        if "ema.pt" in filenames:
            ema_path = os.path.join(dirpath, "ema.pt")
            results.append((dirpath, ema_path))
    
    # Sort by epoch and step
    results.sort(key=lambda x: (
        int(os.path.basename(x[0]).split('-')[0].replace('epoch', '')),
        int(os.path.basename(x[0]).split('-')[1].replace('global_step', ''))
    ))
    return results


def load_model(cfg, ema_path, device, dtype):
    """Load model with weights from checkpoint"""
    logger = create_logger()
    
    # Build VAE model
    vae = build_module(cfg.get("vae", None), MODELS).to(device, dtype).eval()
    
    # Build diffusion model
    image_size = cfg.get("image_size", (240, 240))
    num_frames = cfg.dataset.num_frames
    input_size = (num_frames, *image_size)
    latent_size = vae.get_latent_size(input_size)
    
    # Build text encoder for compatibility
    text_encoder_output_dim = cfg.get("text_encoder_output_dim", 4096)
    text_encoder_model_max_length = cfg.get("text_encoder_model_max_length", 300)
    
    model = build_module(
        cfg.model,
        MODELS,
        input_size=latent_size,
        in_channels=vae.out_channels,
        caption_channels=text_encoder_output_dim,
        model_max_length=text_encoder_model_max_length,
        enable_sequence_parallelism=False,
    ).to(device, dtype).eval()
    
    # Load checkpoint
    logger.info(f"Loading checkpoint from {ema_path}")
    checkpoint = torch.load(ema_path, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    
    return model, vae


def evaluate_checkpoint(cfg, model, vae, dataloader, device, dtype):
    """Evaluate reconstruction loss for a single checkpoint"""
    mae_losses = []
    mse_losses = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            x = batch["video"].to(device, dtype)  # [B, C, T, H, W]
            
            # VAE encode & decode
            z = vae.encode(x)
            x_rec = vae.decode(z, num_frames=x.size(2))
            
            # Calculate losses
            mae_loss = F.l1_loss(x_rec, x)
            mse_loss = F.mse_loss(x_rec, x)
            
            mae_losses.append(mae_loss.item())
            mse_losses.append(mse_loss.item())
    
    avg_mae = sum(mae_losses) / len(mae_losses)
    avg_mse = sum(mse_losses) / len(mse_losses)
    
    return {
        "mae": avg_mae,
        "mse": avg_mse,
        "psnr": -10 * torch.log10(torch.tensor(avg_mse)).item()  # PSNR calculation
    }


def main():
    args = parse_args()
    torch.set_grad_enabled(False)
    
    # Initialize distributed environment if needed
    if torch.cuda.is_available():
        device = torch.device(args.device)
        if is_distributed():
            colossalai.launch_from_torch({})
            coordinator = DistCoordinator()
        else:
            coordinator = None
    else:
        device = torch.device("cpu")
        coordinator = None
    
    # Parse config
    cfg = parse_configs(config_file=args.config, training=False)
    cfg_dtype = cfg.get("dtype", "bf16")
    dtype = to_torch_dtype(cfg_dtype)
    
    # Create logger
    logger = create_logger()
    logger.info("Starting evaluation...")
    
    # Set up validation dataset
    # Modify cfg.dataset with validation path
    cfg.dataset.time_series_dir = os.path.join(args.val_dir, "figure", "360p", "L16-S8")
    cfg.dataset.brightness_dir = os.path.join(args.val_dir, "brightness", "L16-S8")
    
    # Build dataset and dataloader
    dataset = build_module(cfg.dataset, DATASETS)
    logger.info(f"Validation dataset contains {len(dataset)} samples")
    
    dataloader_args = dict(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=cfg.get("seed", 1024),
        shuffle=False,
        drop_last=False,
        pin_memory=True,
    )
    
    dataloader, _ = prepare_dataloader(**dataloader_args)
    
    # Find all checkpoints
    checkpoints = find_checkpoints(args.ckpt_dir)
    logger.info(f"Found {len(checkpoints)} checkpoints to evaluate")
    
    # Results container
    results = []
    
    # Evaluate each checkpoint
    for ckpt_dir, ema_path in checkpoints:
        ckpt_name = os.path.basename(ckpt_dir)
        logger.info(f"Evaluating checkpoint: {ckpt_name}")
        
        # Load model
        model, vae = load_model(cfg, ema_path, device, dtype)
        
        # Evaluate
        metrics = evaluate_checkpoint(cfg, model, vae, dataloader, device, dtype)
        
        # Add to results
        result = {
            "checkpoint": ckpt_name,
            "epoch": int(ckpt_name.split('-')[0].replace('epoch', '')),
            "global_step": int(ckpt_name.split('-')[1].replace('global_step', '')),
            **metrics
        }
        results.append(result)
        
        logger.info(f"Results for {ckpt_name}: MAE: {metrics['mae']:.6f}, MSE: {metrics['mse']:.6f}, PSNR: {metrics['psnr']:.2f} dB")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output, index=False)
    logger.info(f"Results saved to {args.output}")
    
    # Print best checkpoints
    best_mae = results_df.loc[results_df['mae'].idxmin()]
    best_psnr = results_df.loc[results_df['psnr'].idxmax()]
    
    logger.info(f"Best checkpoint by MAE: {best_mae['checkpoint']} (MAE: {best_mae['mae']:.6f})")
    logger.info(f"Best checkpoint by PSNR: {best_psnr['checkpoint']} (PSNR: {best_psnr['psnr']:.2f} dB)")


if __name__ == "__main__":
    main()