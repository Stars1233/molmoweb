#!/bin/bash
set -ex

MIXTURE=molmoweb
WANDB_PROJECT=zixianm_webolmo
WANDB_ENTITY=prior-ai2

CHECKPOINT_PATH=/weka/oe-training-default/sanghol/molmo/models/uber-v1/uber3.4-synthetic-siglip2-qwen3_4b/step30000
CHECKPOINT_ID=4b

DATA_DIR=/weka/oe-training-default/mm-olmo
WEBOLMO_DATA_DIR=/weka/oe-training-default/webolmo/datasets
WEBOLMO_DATASET_VERSION=webolmo_synthetic_june_2_2025

NUM_GPUS=8
NUM_NODES=1
PORT=2041
SEQ_LEN=10240
GLOBAL_BATCH_SIZE=64
DEVICE_BATCH_SIZE=2
DURATION=500
SAVE_INTERVAL=100
EVAL_INTERVAL=$DURATION

current_date_time=$(date +"%m-%d-%H-%M")
RUN_NAME="train_${MIXTURE}_${CHECKPOINT_ID}_${current_date_time}"
SAVE_FOLDER=/weka/oe-training-default/webolmo/zixianm/checkpoints/${RUN_NAME}

NCCL_BLOCKING_WAIT=1 NCCL_TIMEOUT=1800 \
WANDB_ENTITY=${WANDB_ENTITY} WANDB_PROJECT=${WANDB_PROJECT} \
DATA_DIR=${DATA_DIR} MOLMO_DATA_DIR=${DATA_DIR} \
WEBOLMO_DATA_DIR=${WEBOLMO_DATA_DIR} WEBOLMO_DATASET_VERSION=${WEBOLMO_DATASET_VERSION} \
torchrun -m \
  --nproc-per-node ${NUM_GPUS} \
  --nnodes ${NUM_NODES}:${NUM_NODES} \
  --rdzv_backend=c10d \
  --rdzv_id=606 \
  --rdzv_conf="read_timeout=600" \
  --rdzv_endpoint=localhost:${PORT} \
  launch_scripts.train ${MIXTURE} ${CHECKPOINT_PATH} \
  --save_folder=${SAVE_FOLDER} \
  --run_name=${RUN_NAME} \
  --global_batch_size ${GLOBAL_BATCH_SIZE} \
  --device_batch_size ${DEVICE_BATCH_SIZE} \
  --device_eval_batch_size ${DEVICE_BATCH_SIZE} \
  --device_inf_batch_size ${DEVICE_BATCH_SIZE} \
  --duration ${DURATION} \
  --seq_len ${SEQ_LEN} \
  --save_interval ${SAVE_INTERVAL} \
  --eval_interval ${EVAL_INTERVAL} \
  --inf_eval_interval ${EVAL_INTERVAL}
