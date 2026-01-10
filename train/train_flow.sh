#!/bin/bash
# Training script for SpatialVLA with Flow Matching on GPUs 2,3

# Environment Setup
export CUDA_VISIBLE_DEVICES=2,3
export WANDB_PROJECT="spatialvla_flow"

# Paths
# Base path for Model Zoo components (LLaVA, SigLIP, MapAnything)
# Ensure this matches where your models are stored.
export MODEL_ZOO_BASE="/2025233147/mapAnythingLlava3dPi0.5/model_parameters"

# Path to the SpatialVLA Config directory (contains config.json, processor_config.json)
# This config should point to the sub-models in MODEL_ZOO_BASE.
MODEL_CONFIG_PATH="config/spatialvla_dev_integrated"

# Data Paths
# Using glob pattern to load all chunks (000, 001)
DATA_ROOT="/2025233147/zzq/cache/huggingface/lerobot/physical-intelligence/libero/data"
TRAIN_DATA_PATH="${DATA_ROOT}/chunk-*/*.parquet" 
NORM_STATS_PATH="${DATA_ROOT}/norm_stats.json"
TASKS_JSON_PATH="${DATA_ROOT}/../meta/tasks.jsonl"

OUTPUT_DIR="./checkpoints/spatialvla_flow_v1"

# Hyperparameters
NUM_EPOCHS=3
BATCH_SIZE=4 # Per GPU
GRAD_ACCUM=4
LEARNING_RATE=2e-5
ACTION_DIM=19

echo ">>> Starting Training on GPUs $CUDA_VISIBLE_DEVICES..."
echo ">>> Config: $MODEL_CONFIG_PATH"
echo ">>> Data: $TRAIN_DATA_PATH"

# Run with torchrun for DDP
torchrun --nproc_per_node=2 --master_port=29500 train/train_flow.py \
    --model_name_or_path "$MODEL_CONFIG_PATH" \
    --train_data_path "$TRAIN_DATA_PATH" \
    --norm_stats_path "$NORM_STATS_PATH" \
    --tasks_json_path "$TASKS_JSON_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --learning_rate $LEARNING_RATE \
    --lr_scheduler_type "cosine" \
    --warmup_ratio 0.03 \
    --logging_steps 1 \
    --save_steps 500 \
    --save_total_limit 2 \
    --gradient_checkpointing True \
    --bf16 False \
    --tf32 True \
    --dataloader_num_workers 4 \
    --remove_unused_columns False \
    --report_to "none" \
    --do_train \
    --action_dim $ACTION_DIM \
    --ddp_find_unused_parameters True
