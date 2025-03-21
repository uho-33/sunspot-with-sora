import json
import numpy as np
import pandas as pd
import wandb
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

# Initialize wandb
wandb.init(project="sunspot-metrics", name="metrics-heatmap-visualization")

# Path to the detailed metrics file
metrics_file = r"H:\我的云端硬盘\projects\sunspot-with-sora\outputs\evaluation_results_time_2\0009-Sunspot_STDiT3-XL-2\ema.pt\detailed_metrics.json"

# Load the data
print(f"Loading data from {metrics_file}...")
with open(metrics_file, 'r') as f:
    data = json.load(f)

# Extract metrics for heatmaps
metrics_to_plot = ["mae", "mse", "psnr", "ssim"]
sample_indices = set()
steps = []

# First pass to get all unique sample indices and steps
for entry in data:
    steps.append(entry["avg_metrics"]["step"])
    for sample in entry["per_sample_metrics"]:
        sample_indices.add(sample["sample_idx"])

sample_indices = sorted(list(sample_indices))
steps = sorted(steps)

# Create a dataframe for each metric
print("Organizing data for visualization...")
metric_data = {metric: np.zeros((len(steps), len(sample_indices))) for metric in metrics_to_plot}

# Fill in the data
for i, entry in enumerate(tqdm(data)):
    step_idx = steps.index(entry["avg_metrics"]["step"])
    for sample in entry["per_sample_metrics"]:
        sample_idx = sample_indices.index(sample["sample_idx"])
        for metric in metrics_to_plot:
            metric_data[metric][step_idx, sample_idx] = sample["metrics"][metric]

# Log heatmaps to wandb
print("Creating and logging heatmaps to wandb...")
for metric in metrics_to_plot:
    plt.figure(figsize=(12, 8))
    plt.imshow(metric_data[metric], aspect='auto', cmap='viridis')
    plt.colorbar(label=metric)
    plt.xlabel("Sample Index")
    plt.ylabel("Step")
    plt.title(f"{metric.upper()} Heatmap")
    
    # Set x and y ticks (show a subset to avoid overcrowding)
    x_ticks = np.linspace(0, len(sample_indices)-1, min(10, len(sample_indices)), dtype=int)
    y_ticks = np.linspace(0, len(steps)-1, min(10, len(steps)), dtype=int)
    plt.xticks(x_ticks, [sample_indices[i] for i in x_ticks])
    plt.yticks(y_ticks, [steps[i] for i in y_ticks])
    
    wandb.log({f"{metric}_heatmap": wandb.Image(plt)})
    plt.close()

    # Also log as a wandb Table with heatmap
    df = pd.DataFrame(metric_data[metric], 
                     index=[f"Step {step}" for step in steps],
                     columns=[f"Sample {idx}" for idx in sample_indices])
    
    wandb_table = wandb.Table(dataframe=df)
    wandb.log({f"{metric}_table": wandb_table})

print("Done! Check your wandb dashboard for the heatmaps.")
wandb.finish()
