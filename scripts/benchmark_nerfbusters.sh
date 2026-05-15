#!/usr/bin/env bash

export PYTHONDONTWRITEBYTECODE=1

CODE_ROOT_DIR=$HOME/'Projects/freescale/gaussian_splatting'
cd $CODE_ROOT_DIR

DATASET_PATH=${HOME}/datasets/nerfbusters
# # downsample factor = 2
# CSV_PATH=/home/yuchen/datasets/nerfbusters/nerfbusters2.csv
# downsample factor = 4
CSV_PATH=/home/yuchen/datasets/nerfbusters/nerfbusters1.csv

EXP_OUT_DIR=${HOME}/datasets/nerfbusters/out
mkdir -p $EXP_OUT_DIR

ACTIVE_EXP_OUT_DIR=${HOME}/datasets/nerfbusters/out_active
mkdir -p $ACTIVE_EXP_OUT_DIR

START_INDEX=0
END_INDEX=-1
SUFFIX=3dgs

python -m scripts.train.neural_mapping_dispatch $DATASET_PATH \
    --csv_path $CSV_PATH \
    --start_index $START_INDEX \
    --end_index $END_INDEX \
    --suffix $SUFFIX \
    --run_3dgs \
    --exp_out_dir $EXP_OUT_DIR \
    --active_exp_out_dir $ACTIVE_EXP_OUT_DIR \
    --active_training \
    --train_multi_gpus \
    --num_gpus 4
