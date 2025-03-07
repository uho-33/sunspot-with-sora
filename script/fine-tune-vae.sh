#!/bin/bash

# Set base directories
BASE_DIR="/content"
PROJECT_DIR="${BASE_DIR}/drive/MyDrive/projects/sunspot-with-sora"
DATA_DIR="${BASE_DIR}/dataset"
LOG_DIR="${PROJECT_DIR}/log"

# Create necessary directories
mkdir -p "${LOG_DIR}"
mkdir -p "${PROJECT_DIR}/checkpoint"

# Check if directories exist
if [ ! -d "${DATA_DIR}/training/time-series/360p/L16-S8" ]; then
    echo "Error: Training data directory not found"
    exit 1
fi

if [ ! -d "${DATA_DIR}/validation/time-series/360p/L16-S8" ]; then
    echo "Warning: Validation data directory not found"
fi

# Set environment variable to enable expandable segments
# This helps with GPU memory fragmentation issues
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run the training script with reduced memory requirements
python Fine_tune/vae/finetune_vae.py \
    --data_dir "${DATA_DIR}/training/time-series/360p/L16-S8" \
    --val_dir "${DATA_DIR}/validation/time-series/360p/L16-S8" \
    --pretrained_path "saved_models/opensora_vae_v1.3.safetensors" \
    --output_dir "${PROJECT_DIR}/checkpoint" \
    --sequence_length 16 \
    --image_size 160 \
    --batch_size 1 \
    --epochs 50 \
    --lr 1e-5 \
    --beta 0.001 \
    --micro_frame_size 4 \
    --wandb_project "sunspot-vae" \
    --run_name "finetune_vae_lower_res" \
    --save_interval 5 \
    --validate_interval 1 \
    --log_interval 5 \
    --early_stopping 10 \
    --lr_patience 3 \
    --lr_factor 0.5 \
    >& "${LOG_DIR}/fine-tune.log"

# Check if the training completed successfully
if [ $? -eq 0 ]; then
    echo "Training completed successfully"
    echo "Log file: ${LOG_DIR}/fine-tune.log"
else
    echo "Training failed"
    echo "Check log file: ${LOG_DIR}/fine-tune.log"
    exit 1
fi