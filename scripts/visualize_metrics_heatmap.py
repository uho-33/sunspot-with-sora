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
metrics_file = r"H:\我的云端硬盘\projects\sunspot-with-sora\outputs\evaluation\time_dependency\detailed_metrics.json"

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

# Extract timestamps for each sample
timestamps = []
for i, entry in enumerate(data):
    if i == 0:  # Just use the first entry to get timestamps
        for sample in entry["per_sample_metrics"]:
            timestamps.append((sample["sample_idx"], sample["timestamp"]))

timestamps.sort(key=lambda x: x[0])  # Sort by sample_idx
sample_to_timestamp = {sample_idx: timestamp for sample_idx, timestamp in timestamps}
timestamp_values = [t[1] for t in timestamps]

# Fill in the data
for i, entry in enumerate(tqdm(data)):
    step_idx = steps.index(entry["avg_metrics"]["step"])
    for sample in entry["per_sample_metrics"]:
        sample_idx = sample_indices.index(sample["sample_idx"])
        for metric in metrics_to_plot:
            metric_data[metric][step_idx, sample_idx] = sample["metrics"][metric]

# 创建按timestamp排序的样本索引
timestamp_sorted_indices = sorted(range(len(sample_indices)), 
                                key=lambda i: sample_to_timestamp[sample_indices[i]])
timestamp_sorted_samples = [sample_indices[i] for i in timestamp_sorted_indices]
timestamp_sorted_values = [sample_to_timestamp[idx] for idx in timestamp_sorted_samples]

# Log heatmaps to wandb using matplotlib
print("Creating and logging heatmaps to wandb...")
for metric in metrics_to_plot:
    # 1. 创建以sample_idx为x轴的热图（小值亮色，大值暗色）
    plt.figure(figsize=(12, 8))
    plt.imshow(metric_data[metric], aspect='auto', cmap='viridis_r')  # 使用反向颜色映射
    plt.colorbar(label=metric.upper())
    plt.xlabel("Sample Index")
    plt.ylabel("Training Step")
    plt.title(f"{metric.upper()} Heatmap (Sample Index as X-axis)")
    
    # Set x and y ticks (show a subset to avoid overcrowding)
    x_ticks = np.linspace(0, len(sample_indices)-1, min(10, len(sample_indices)), dtype=int)
    y_ticks = np.linspace(0, len(steps)-1, min(10, len(steps)), dtype=int)
    plt.xticks(x_ticks, [sample_indices[i] for i in x_ticks])
    plt.yticks(y_ticks, [steps[i] for i in y_ticks])
    
    # Log the matplotlib figure as an image
    wandb.log({f"{metric.upper()} Heatmap (Sample Index as X)": wandb.Image(plt)})
    plt.close()
    
    # 2. 创建以timestamp为x轴的热图
    plt.figure(figsize=(14, 8))
    # 重新组织数据，按timestamp排序
    timestamp_sorted_data = metric_data[metric][:, timestamp_sorted_indices]
    plt.imshow(timestamp_sorted_data, aspect='auto', cmap='viridis_r')  # 使用反向颜色映射
    plt.colorbar(label=metric.upper())
    plt.xlabel("Sample (sorted by timestamp)")
    plt.ylabel("Training Step")
    plt.title(f"{metric.upper()} Heatmap (Timestamp as X-axis)")
    
    # 设置x轴标签为时间戳值
    x_ticks = np.linspace(0, len(timestamp_sorted_samples)-1, min(15, len(timestamp_sorted_samples)), dtype=int)
    y_ticks = np.linspace(0, len(steps)-1, min(10, len(steps)), dtype=int)
    plt.xticks(x_ticks, [f"{timestamp_sorted_values[i]:.2f}" for i in x_ticks], rotation=45)
    plt.yticks(y_ticks, [steps[i] for i in y_ticks])
    plt.tight_layout()
    
    # Log the timestamp-based heatmap
    wandb.log({f"{metric.upper()} Heatmap (Timestamp as X)": wandb.Image(plt)})
    plt.close()
    
    # 3. 创建以sample为y轴、step为x轴的热图（转置版本）
    plt.figure(figsize=(12, 10))
    plt.imshow(metric_data[metric].T, aspect='auto', cmap='viridis_r')  # 使用反向颜色映射
    plt.colorbar(label=metric.upper())
    plt.xlabel("Training Step")
    plt.ylabel("Sample Index")
    plt.title(f"{metric.upper()} Heatmap (Steps as X, Samples as Y)")
    
    # 设置轴标签
    y_ticks = np.linspace(0, len(sample_indices)-1, min(20, len(sample_indices)), dtype=int)
    x_ticks = np.linspace(0, len(steps)-1, min(10, len(steps)), dtype=int)
    plt.yticks(y_ticks, [f"{sample_indices[i]}" for i in y_ticks])
    plt.xticks(x_ticks, [steps[i] for i in x_ticks])
    plt.xticks(rotation=45)
    
    # 记录这个转置视图
    wandb.log({f"{metric.upper()} Heatmap (Transposed)": wandb.Image(plt)})
    plt.close()
    
    # 4. 创建以timestamp排序的样本为y轴、step为x轴的热图
    plt.figure(figsize=(12, 10))
    plt.imshow(timestamp_sorted_data.T, aspect='auto', cmap='viridis_r')  # 使用反向颜色映射
    plt.colorbar(label=metric.upper())
    plt.xlabel("Training Step")
    plt.ylabel("Sample (sorted by timestamp)")
    plt.title(f"{metric.upper()} Heatmap (Steps as X, Timestamp-sorted Samples as Y)")
    
    # 设置轴标签
    y_ticks = np.linspace(0, len(timestamp_sorted_samples)-1, min(20, len(timestamp_sorted_samples)), dtype=int)
    x_ticks = np.linspace(0, len(steps)-1, min(10, len(steps)), dtype=int)
    plt.yticks(y_ticks, [f"{timestamp_sorted_samples[i]} ({timestamp_sorted_values[i]:.2f})" for i in y_ticks])
    plt.xticks(x_ticks, [steps[i] for i in x_ticks])
    plt.xticks(rotation=45)
    
    # 记录这个基于时间戳排序的转置视图
    wandb.log({f"{metric.upper()} Heatmap (Timestamp-sorted, Transposed)": wandb.Image(plt)})
    plt.close()
    
    # Also log metrics over time as a line plot
    avg_metrics_per_step = np.mean(metric_data[metric], axis=1)
    data_table = wandb.Table(data=[[step, avg_val] for step, avg_val in zip(steps, avg_metrics_per_step)], 
                             columns=["step", f"avg_{metric}"])
    wandb.log({f"{metric}_over_time": wandb.plot.line(
        data_table, "step", f"avg_{metric}", 
        title=f"Average {metric.upper()} vs Training Step"
    )})

print("Done! Check your wandb dashboard for the heatmaps.")
wandb.finish()
