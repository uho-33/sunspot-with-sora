#!/bin/bash

# Set base directories
BASE_DIR="/content"
PROJECT_DIR="${BASE_DIR}/drive/MyDrive/projects/sunspot-with-sora"
DATA_DIR="${BASE_DIR}/dataset"
LOG_DIR="${PROJECT_DIR}/log"
CHECKPOINT_DIR="${PROJECT_DIR}/checkpoint"

# Create necessary directories
mkdir -p "${LOG_DIR}"
mkdir -p "${CHECKPOINT_DIR}"

# Only run initialization script if the dataset directory doesn't exist
if [ ! -d "${DATA_DIR}" ]; then
    echo "Dataset directory not found. Running initialization script..."
    source script/init.sh
else
    echo "Dataset directory found. Skipping initialization."
fi

# Check if directories exist
if [ ! -d "${DATA_DIR}/training/time-series/360p/L16-S8" ]; then
    echo "Error: Training data directory not found"
    exit 1
fi

if [ ! -d "${DATA_DIR}/validation/time-series/360p/L16-S8" ]; then
    echo "Warning: Validation data directory not found"
fi

# Find the latest checkpoint
LATEST_CHECKPOINT=""
if [ -d "${CHECKPOINT_DIR}" ]; then
    # Look for files matching vae_epoch_*.pt and find the most recent one by epoch number
    LATEST_CHECKPOINT=$(find "${CHECKPOINT_DIR}" -name "vae_epoch_*.pt" | sort -V | tail -n 1)
    
    # If best model exists and no epoch checkpoints found, use the best model
    if [ -z "${LATEST_CHECKPOINT}" ] && [ -f "${CHECKPOINT_DIR}/vae_best_model.pt" ]; then
        LATEST_CHECKPOINT="${CHECKPOINT_DIR}/vae_best_model.pt"
    fi
    
    # If a checkpoint was found
    if [ -n "${LATEST_CHECKPOINT}" ]; then
        echo "Found latest checkpoint: ${LATEST_CHECKPOINT}"
    else
        echo "No existing checkpoints found. Will start from scratch."
    fi
fi

# Set the pretrained_path argument based on whether we found a checkpoint
PRETRAINED_ARG=""
if [ -n "${LATEST_CHECKPOINT}" ]; then
    PRETRAINED_ARG="--pretrained_path ${LATEST_CHECKPOINT}"
fi

# Set environment variable to enable expandable segments
# This helps with GPU memory fragmentation issues
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run the training script with the latest checkpoint
echo "Starting training at $(date)"
python Fine_tune/vae/finetune_vae.py \
    --data_dir "${DATA_DIR}/training/time-series/360p/L16-S8" \
    --val_dir "${DATA_DIR}/validation/time-series/360p/L16-S8" \
    ${PRETRAINED_ARG} \
    --output_dir "${CHECKPOINT_DIR}" \
    --sequence_length 16 \
    --image_size 240 \
    --batch_size 1 \
    --epochs 50 \
    --lr 1e-5 \
    --beta 0.001 \
    --micro_frame_size 2 \
    --wandb_project "sunspot-vae" \
    --run_name "finetune_vae_$(date +%Y%m%d_%H%M%S)" \
    --save_interval 5 \
    --validate_interval 1 \
    --log_interval 10 \
    --early_stopping 10 \
    --lr_patience 3 \
    --lr_factor 0.5 \
    --gradient_accumulation_steps 16 \
    --use_amp \
    --amp_dtype "bfloat16" \
    --dtype "bfloat16" \
    --max_grad_norm 1.0 \
    --num_workers 0 \
    --pin_memory \
    >& "${LOG_DIR}/fine-tune_$(date +%Y%m%d_%H%M%S).log"

# Check if the training completed successfully
if [ $? -eq 0 ]; then
    echo "Training completed successfully at $(date)"
    echo "Log file: ${LOG_DIR}/fine-tune_$(date +%Y%m%d_%H%M%S).log"
else
    echo "Training failed at $(date)"
    echo "Check most recent log file in: ${LOG_DIR}"
    exit 1
fi