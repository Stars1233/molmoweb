#!/bin/bash
set -ex

CHECKPOINT_PATH=/weka/oe-training-default/webolmo/zixianm/checkpoints/train_hero_4b_03-14-06-17/step50000

MIXTURE=screenspot:test,screenspot_v2:test

DATA_DIR=/weka/oe-training-default/mm-olmo
WEBOLMO_DATA_DIR=/weka/oe-training-default/webolmo/datasets
WEBOLMO_DATASET_VERSION=webolmo_synthetic_june_2_2025

NUM_GPUS=1
DEVICE_BATCH_SIZE=2

if [[ $CHECKPOINT_PATH == *"step"* ]]; then
  FLATTENED_NAME=$(basename $(dirname "$CHECKPOINT_PATH"))_$(basename "$CHECKPOINT_PATH")
elif [[ $CHECKPOINT_PATH == *"/"* ]] && [[ ! $CHECKPOINT_PATH == /* ]]; then
  FLATTENED_NAME=$(echo $CHECKPOINT_PATH | tr '/' '_')
else
  FLATTENED_NAME=$(basename "$CHECKPOINT_PATH")
fi

SAVE_FOLDER=/weka/oe-training-default/zixianm/molmoweb-internal/results/eval_${FLATTENED_NAME}



DATA_DIR=${DATA_DIR} MOLMO_DATA_DIR=${DATA_DIR} \
WEBOLMO_DATA_DIR=${WEBOLMO_DATA_DIR} WEBOLMO_DATASET_VERSION=${WEBOLMO_DATASET_VERSION} \
torchrun -m \
  --nproc-per-node ${NUM_GPUS} \
  launch_scripts.eval ${CHECKPOINT_PATH} ${MIXTURE} \
  --save_dir=${SAVE_FOLDER} \
  --device_batch_size ${DEVICE_BATCH_SIZE} \
  --include_image
