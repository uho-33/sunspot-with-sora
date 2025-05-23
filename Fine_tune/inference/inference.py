import os
import re
from pprint import pformat
import logging

import colossalai
import torch
import torch.distributed as dist
import torch.nn.functional as F
from colossalai.cluster import DistCoordinator
from mmengine.runner import set_random_seed
from tqdm import tqdm
import numpy as np
import tempfile
import pandas as pd
import os
from os import path as osp

import sys
# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../origin_opensora')))

from opensora.acceleration.parallel_states import get_data_parallel_group
from opensora.datasets.dataloader import prepare_dataloader
from opensora.acceleration.parallel_states import set_sequence_parallel_group
from opensora.datasets.aspect import get_image_size, get_num_frames
from opensora.registry import MODELS, SCHEDULERS, DATASETS, build_module
from opensora.utils.config_utils import parse_configs
from opensora.utils.inference_utils import (
    load_prompts,
    print_memory_usage,
   
)
from opensora.utils.misc import all_exists, create_logger, is_distributed, is_main_process, to_torch_dtype

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from Fine_tune.utils.inference_utils import (
    modified_get_save_path_name, 
    save_video_as_frames,
    load_brightness_prompts,
    process_brightness_data
    )


def main():
    torch.set_grad_enabled(False)
    # ======================================================
    # configs & runtime variables
    # ======================================================
    # == parse configs ==
    cfg = parse_configs(training=False)

    # == device and dtype ==
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg_dtype = cfg.get("dtype", "fp32")
    assert cfg_dtype in ["fp16", "bf16", "fp32"], f"Unknown mixed precision {cfg_dtype}"
    dtype = to_torch_dtype(cfg.get("dtype", "bf16"))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # == init distributed env ==
    if is_distributed():
        colossalai.launch_from_torch({})
        coordinator = DistCoordinator()
        enable_sequence_parallelism = coordinator.world_size > 1
        if enable_sequence_parallelism:
            set_sequence_parallel_group(dist.group.WORLD)
    else:
        coordinator = None
        enable_sequence_parallelism = False
    set_random_seed(seed=cfg.get("seed", 1024))

    # == init logger ==
    logger = create_logger()
    logger.info("Inference configuration:\n %s", pformat(cfg.to_dict()))
    verbose = cfg.get("verbose", 1)
    progress_wrap = tqdm if verbose == 1 else (lambda x: x)

    # ======================================================
    # build model & load weights
    # ======================================================
    logger.info("Building models...")
    # == build text-encoder and vae ==
    text_encoder = build_module(cfg.text_encoder, MODELS, device=device)
    vae = build_module(cfg.vae, MODELS).to(device, dtype).eval()
    is_validation = cfg.get("is_validation", False)
    # == prepare video size ==
    image_size = cfg.get("image_size", None)
    if image_size is None:
        resolution = cfg.get("resolution", None)
        aspect_ratio = cfg.get("aspect_ratio", None)
        assert (
            resolution is not None and aspect_ratio is not None
        ), "resolution and aspect_ratio must be provided if image_size is not provided"
        image_size = get_image_size(resolution, aspect_ratio)
    num_frames = get_num_frames(cfg.num_frames)

    # Set FPS for generation in non-validation mode
    fps = cfg.get("fps", 8)  # Default to 8 if not specified

    # == build diffusion model ==
    input_size = (num_frames, *image_size)
    latent_size = vae.get_latent_size(input_size)
    model = (
        build_module(
            cfg.model,
            MODELS,
            input_size=latent_size,
            in_channels=vae.out_channels,
            caption_channels=text_encoder.output_dim,
            model_max_length=text_encoder.model_max_length,
            enable_sequence_parallelism=enable_sequence_parallelism,
        )
        .to(device, dtype)
        .eval()
    )
    text_encoder.y_embedder = model.y_embedder  # HACK: for classifier-free guidance
    
    # Patch for y_embedder null method in fourier text encoder if needed
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

    # == build scheduler ==
    scheduler = build_module(cfg.scheduler, SCHEDULERS)

    # ======================================================
    # inference
    # ======================================================
    # == load prompts ==
    prompts = cfg.get("prompt", None)
    if prompts is not None and not isinstance(prompts, torch.Tensor):
        prompts = torch.tensor(prompts)
    start_idx = cfg.get("start_index", 0)
    
    if prompts is None and not is_validation: 
        if cfg.get("prompt_path", None) is not None:
            prompts, prompt_filename  = load_brightness_prompts(cfg.prompt_path, start_idx, cfg.get("end_index", None))
        else:
            print("Warning: no input brightness data, use 0 series as data")
            prompts = torch.tensor([[0]]) 
    process_len = len(prompts)

    if is_validation:
        logger.info("Building dataset...")
        # == build dataset ==
        dataset = build_module(cfg.dataset, DATASETS)
        logger.info("Dataset contains %s samples.", len(dataset))

        # == build dataloader ==
        dataloader_args = dict(
            dataset=dataset,
            batch_size=cfg.get("batch_size", 1),
            num_workers=cfg.get("num_workers", 4),
            seed=cfg.get("seed", 1024),
            shuffle=False,
            drop_last=True,
            pin_memory=True,
            process_group=get_data_parallel_group(),
            prefetch_factor=cfg.get("prefetch_factor", None),
        )
        dataloader, _ = prepare_dataloader(
            bucket_config=cfg.get("bucket_config", None),
            num_bucket_build_workers=cfg.get("num_bucket_build_workers", 1),
            **dataloader_args,
        )
        dataloader_iter = iter(dataloader)
        process_len = len(dataset)


    # == prepare arguments ==
    batch_size = cfg.get("batch_size", 1)
    num_sample = cfg.get("num_sample", 1)

    save_dir = cfg.save_dir
    os.makedirs(save_dir, exist_ok=True)
    sample_name = cfg.get("sample_name", None)
    prompt_as_path = cfg.get("prompt_as_path", False)

    prompt_filename_as_path = True  # Force it to be True regardless of config

    use_sdedit = cfg.get("use_sdedit", False)
    use_oscillation_guidance_for_text = cfg.get("use_oscillation_guidance_for_text", None)
    use_oscillation_guidance_for_image = cfg.get("use_oscillation_guidance_for_image", None)

    
    # == Iter over all samples ==
    for i in progress_wrap(range(0, process_len, batch_size)):
        # == prepare batch prompts ==
        if is_validation:
            batch = next(dataloader_iter)
            prompt_filename = batch['file_name'] if prompt_filename_as_path else None
            original_batch_prompts = batch['text']
            # Process brightness/text data
            batch_prompts = process_brightness_data(original_batch_prompts, logger)
        else:
            batch_prompts = prompts[i : i + batch_size]
            original_batch_prompts = batch_prompts
        
        
        # == Iter over number of sampling for one prompt ==
        for k in range(num_sample):
            # == prepare save paths ==
            save_paths = [
                modified_get_save_path_name(
                    save_dir,
                    sample_name=sample_name,
                    sample_idx=start_idx + idx,
                    prompt=original_batch_prompts[idx],
                    prompt_as_path=prompt_as_path,
                    prompt_filename_as_path=prompt_filename_as_path,
                    prompt_filename=prompt_filename[idx] if prompt_filename_as_path else None,
                    num_sample=num_sample,
                    k=k,
                )
                for idx in range(len(batch_prompts))
            ]

            # NOTE: Skip if the sample already exists
            # This is useful for resuming sampling VBench
            if (prompt_as_path or prompt_filename_as_path) and all_exists(save_paths):
                continue

            video_clips = []
            # == sampling ==
            torch.manual_seed(1024)
            z = torch.randn(len(batch_prompts), vae.out_channels, *latent_size, device=device, dtype=dtype)

            try:
                # Always create model_kwargs with proper dimensions
                batch_size = len(batch_prompts)
                
                # For validation data, use dimensions from batch
                if is_validation and 'height' in batch and 'width' in batch and 'num_frames' in batch:
                    model_kwargs = {
                        "height": batch["height"].to(device),
                        "width": batch["width"].to(device),
                        "num_frames": batch["num_frames"].to(device),
                        "fps": batch["fps"].to(device) if "fps" in batch else torch.ones(batch_size, device=device) * 16.0
                    }
                else:
                    # For non-validation mode or when batch doesn't have dimensions,
                    # set parameters from config - ensure values are at least 1
                    model_kwargs = {
                        "height": torch.tensor([max(1, image_size[0])] * batch_size, device=device),
                        "width": torch.tensor([max(1, image_size[1])] * batch_size, device=device),
                        "num_frames": torch.tensor([max(1, num_frames)] * batch_size, device=device),
                        "fps": torch.tensor([fps] * batch_size, device=device),
                    }
                
                # Debug log the model_kwargs to verify dimensions
                if verbose >= 2:
                    logger.info(f"Using model_kwargs: {model_kwargs}")
                    logger.info(f"z shape: {z.shape}, batch_prompts shape: {batch_prompts.shape if isinstance(batch_prompts, torch.Tensor) else 'not a tensor'}")
                
                # Ensure latent size is compatible with the model
                # if z.shape[2] == 0 or z.shape[3] == 0 or z.shape[4] == 0:
                #     logger.error(f"Invalid latent dimensions: {z.shape}. Adjusting...")
                #     # Fix dimensions to prevent zero-sized dimensions
                #     if z.shape[2] == 0:  # Time dimension
                #         z = torch.randn(len(batch_prompts), vae.out_channels, 1, z.shape[3], z.shape[4], device=device, dtype=dtype)
                #     if z.shape[3] == 0 or z.shape[4] == 0:  # Height or width
                #         min_spatial_dim = 32  # Minimum spatial dimension for the model
                #         z = torch.randn(len(batch_prompts), vae.out_channels, max(1, z.shape[2]), 
                #                        max(min_spatial_dim, z.shape[3]), max(min_spatial_dim, z.shape[4]), 
                #                        device=device, dtype=dtype)

                samples = scheduler.sample(
                    model,
                    text_encoder,
                    z=z,
                    prompts=batch_prompts,
                    device=device,
                    progress=verbose >= 2,
                    use_sdedit=use_sdedit,
                    use_oscillation_guidance_for_text=use_oscillation_guidance_for_text,
                    use_oscillation_guidance_for_image=use_oscillation_guidance_for_image,
                    additional_args=model_kwargs  # Pass dimensions to the model
                )
                video_clips.append(samples)
                print(f"video_clips[0].shape :{video_clips[0].shape}")
            except Exception as e:
                logger.error(f"Error during sampling: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                logger.error("Skipping this batch due to error")
                continue  # Skip to the next batch

            # == save samples ==
            if is_main_process():
                for idx, batch_prompt in enumerate(batch_prompts):
                    try:
                        if verbose >= 2:
                            logger.info("Prompt: %s", batch_prompt)
                        save_path = save_paths[idx]
                        video = [video_clips[0][idx]]  # Access the first and only element from video_clips
                        video = torch.cat(video, dim=1)  # latent [C, T, H, W]
                        # ensure latent frame size is multiples of 5
                        t_cut = video.size(1) // 5 * 5
                        if t_cut < video.size(1):
                            video = video[:, :t_cut]


                        # Decode with proper error handling
                        try:
                            decoded_video = vae.decode(video.to(dtype), num_frames=t_cut * 17 // 5).squeeze(0)
                            logger.info(f"Video decoded successfully, size: {decoded_video.size()}")
                            print(f"Video decoded successfully, size: {decoded_video.size()}")
                            
                            save_path = save_video_as_frames(
                                decoded_video, 
                                save_dir=save_path,
                                verbose=verbose >= 2,
                            )
                            logger.info(f"Video saved to {save_path}")
                        except Exception as e:
                            logger.error(f"Error during video decoding or saving: {e}")
                            import traceback
                            logger.error(f"Traceback: {traceback.format_exc()}")
                    except Exception as e:
                        logger.error(f"Error processing sample {idx}: {e}")
                    
        start_idx += len(batch_prompts)
    logger.info("Inference finished.")
    logger.info("Saved %s samples to %s", start_idx, save_dir)
    print_memory_usage("After inference", device)


if __name__ == "__main__":
    main()