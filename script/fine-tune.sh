source script/init.sh
torchrun --standalone --nproc_per_node 1 origin_opensora/scripts/train.py \
    Fine_tune/configs/train/fine_tune_stage1.py