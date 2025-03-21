#!/bin/bash

# Script to run time-dependent evaluation across all checkpoints

# Default values
CONFIG="Fine_tune/configs/train/evaluate.py"
CHECKPOINTS_DIR="outputs/heatmap_ckp"
VALIDATION_DATA_DIR="/content/dataset/validation"
RESULTS_DIR="outputs/evaluation/time_dependency"
BATCH_SIZE=80
USE_WANDB=True
WANDB_PROJECT="sun-reconstruction-time-eval"
WANDB_ENTITY=""
WANDB_RUN_NAME="time-dependency-heatmap-ckp"
NUM_GPUS=1
STEP_INTERVAL=600  # Sample checkpoints every 600 steps
DISABLE_BNB=true  # Disable bitsandbytes by default to avoid library issues

# Parse named arguments
while [ $# -gt 0 ]; do
  case "$1" in
    --config=*)
      CONFIG="${1#*=}"
      ;;
    --checkpoints_dir=*)
      CHECKPOINTS_DIR="${1#*=}"
      ;;
    --validation_data_dir=*)
      VALIDATION_DATA_DIR="${1#*=}"
      ;;
    --results_dir=*)
      RESULTS_DIR="${1#*=}"
      ;;
    --batch_size=*)
      BATCH_SIZE="${1#*=}"
      ;;
    --use_wandb)
      USE_WANDB=true
      ;;
    --no_wandb)
      USE_WANDB=false
      ;;
    --wandb_project=*)
      WANDB_PROJECT="${1#*=}"
      ;;
    --wandb_entity=*)
      WANDB_ENTITY="${1#*=}"
      ;;
    --wandb_run_name=*)
      WANDB_RUN_NAME="${1#*=}"
      ;;
    --num_gpus=*)
      NUM_GPUS="${1#*=}"
      ;;
    --step_interval=*)
      STEP_INTERVAL="${1#*=}"
      ;;
    --disable_bnb)
      DISABLE_BNB=true
      ;;
    --enable_bnb)
      DISABLE_BNB=false
      ;;
    *)
      echo "Unknown parameter: $1"
      exit 1
      ;;
  esac
  shift
done

echo "Running time-dependent evaluation with the following parameters:"
echo "Config: $CONFIG"
echo "Checkpoints directory: $CHECKPOINTS_DIR"
echo "Validation data directory: $VALIDATION_DATA_DIR"
echo "Results directory: $RESULTS_DIR"
echo "Batch size: $BATCH_SIZE"
echo "Number of GPUs: $NUM_GPUS"
echo "Step interval: $STEP_INTERVAL"
echo "Disable bitsandbytes: $DISABLE_BNB"
echo "Use wandb: $USE_WANDB"
if [ "$USE_WANDB" = true ]; then
  echo "Wandb project: $WANDB_PROJECT"
  echo "Wandb entity: $WANDB_ENTITY"
  echo "Wandb run name: $WANDB_RUN_NAME"
fi

# Create results directory if it doesn't exist
mkdir -p "$RESULTS_DIR"

# Export environment variables to disable bitsandbytes
if [ "$DISABLE_BNB" = true ]; then
  export BITSANDBYTES_NOWELCOME=1
  export BNB_DISABLE_QUANTIZATION=1
  echo "Disabled bitsandbytes through environment variables"
fi

# Build base command with properly escaped parameters
CMD="torchrun --standalone --nproc_per_node=$NUM_GPUS Fine_tune/eval/evaluate_checkpoints_time_dependency.py"
CMD="$CMD --config=\"$CONFIG\""
CMD="$CMD --checkpoints_dir=\"$CHECKPOINTS_DIR\""
CMD="$CMD --validation_data_dir=\"$VALIDATION_DATA_DIR\""
CMD="$CMD --results_dir=\"$RESULTS_DIR\""
CMD="$CMD --step_interval=$STEP_INTERVAL"
CMD="$CMD --batch_size=$BATCH_SIZE"

# Add bitsandbytes flag
if [ "$DISABLE_BNB" = true ]; then
  CMD="$CMD --disable_bnb"
fi

# Add wandb parameters if enabled - add each parameter separately
if [ "$USE_WANDB" = true ]; then
  CMD="$CMD --use_wandb"
  CMD="$CMD --wandb_project=\"$WANDB_PROJECT\""
  
  if [ ! -z "$WANDB_ENTITY" ]; then
    CMD="$CMD --wandb_entity=\"$WANDB_ENTITY\""
  fi
  
  if [ ! -z "$WANDB_RUN_NAME" ]; then
    CMD="$CMD --wandb_run_name=\"$WANDB_RUN_NAME\""
  fi
  
  # Debug output to verify wandb parameters
  echo "WANDB_PROJECT=$WANDB_PROJECT"
  echo "WANDB_RUN_NAME=$WANDB_RUN_NAME"
fi

# Debug: print the constructed command for verification
echo "Debug: Full command to be executed:"
echo "$CMD"

# Execute the command
echo "Running command: $CMD"
eval $CMD

echo "Time-dependent evaluation completed. Results saved to $RESULTS_DIR"
