#!/bin/bash

# Script to run the evaluation across all checkpoints

# Default values
CONFIG="Fine_tune/configs/train/fine_tune_stage1.py"
CHECKPOINTS_DIR="outputs/0002-Sunspot_STDiT3-XL-2"
VALIDATION_DATA_DIR="/content/dataset/validation"
RESULTS_DIR="evaluation_results"
BATCH_SIZE=4
USE_WANDB=True
WANDB_PROJECT="sun-reconstruction-eval"
WANDB_ENTITY=""
WANDB_RUN_NAME="stage1"

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
    --wandb_project=*)
      WANDB_PROJECT="${1#*=}"
      ;;
    --wandb_entity=*)
      WANDB_ENTITY="${1#*=}"
      ;;
    --wandb_run_name=*)
      WANDB_RUN_NAME="${1#*=}"
      ;;
    *)
      echo "Unknown parameter: $1"
      exit 1
      ;;
  esac
  shift
done

echo "Running evaluation with the following parameters:"
echo "Config: $CONFIG"
echo "Checkpoints directory: $CHECKPOINTS_DIR"
echo "Validation data directory: $VALIDATION_DATA_DIR"
echo "Results directory: $RESULTS_DIR"
echo "Batch size: $BATCH_SIZE"
echo "Use wandb: $USE_WANDB"
if [ "$USE_WANDB" = true ]; then
  echo "Wandb project: $WANDB_PROJECT"
  echo "Wandb entity: $WANDB_ENTITY"
  echo "Wandb run name: $WANDB_RUN_NAME"
fi

# Create results directory if it doesn't exist
mkdir -p "$RESULTS_DIR"

# Build command
CMD="python evaluate_checkpoints.py \
  --config=\"$CONFIG\" \
  --checkpoints_dir=\"$CHECKPOINTS_DIR\" \
  --validation_data_dir=\"$VALIDATION_DATA_DIR\" \
  --results_dir=\"$RESULTS_DIR\" \
  --batch_size=\"$BATCH_SIZE\""

# Add wandb parameters if enabled
if [ "$USE_WANDB" = true ]; then
  CMD="$CMD --use_wandb --wandb_project=\"$WANDB_PROJECT\""
  
  if [ ! -z "$WANDB_ENTITY" ]; then
    CMD="$CMD --wandb_entity=\"$WANDB_ENTITY\""
  fi
  
  if [ ! -z "$WANDB_RUN_NAME" ]; then
    CMD="$CMD --wandb_run_name=\"$WANDB_RUN_NAME\""
  fi
fi

# Execute the command
eval $CMD

echo "Evaluation completed. Results saved to $RESULTS_DIR"
