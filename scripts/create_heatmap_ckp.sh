#!/bin/bash

# 配置路径
src_dir="/content/drive/MyDrive/projects/sunspot-with-sora/outputs/0001-Sunspot_STDiT3-XL-2"
dst_dir="/content/drive/MyDrive/projects/sunspot-with-sora/outputs/heatmap_ckp"
offset=2000
interval=8

# 创建目标目录
mkdir -p "$dst_dir"

# 提取所有包含 global_step 的目录的 step 值
mapfile -t steps < <(
  find "$src_dir" -type d -name "*global_step*" \
  -exec basename {} \; \
  | grep -oP 'global_step\d+' \
  | grep -oP '\d+' \
  | sort -n | uniq
)

# 按间隔取样 step 值
for ((i=0; i<${#steps[@]}; i+=interval)); do
    step=${steps[i]}
    folder=$(find "$src_dir" -type d -name "*global_step${step}*" | head -n 1)

    if [ -d "$folder" ]; then
        ema_file="$folder/ema.pt"
        if [ -f "$ema_file" ]; then
            new_step=$((step + offset))
            new_name="step${new_step}.pt"
            cp "$ema_file" "$dst_dir/$new_name"
            echo "✅ Copied $ema_file -> $dst_dir/$new_name"
        else
            echo "⚠️ Warning: ema.pt not found in $folder"
        fi
    else
        echo "⚠️ Warning: Folder not found for step $step"
    fi
done
