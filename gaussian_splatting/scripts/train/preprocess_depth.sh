#!/usr/bin/env bash

CUDA_IDS=$1 # {'0,1,2,...'}
DATASET_PATH=$2

export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=${CUDA_IDS}

CODE_ROOT_DIR='/cephyr/users/qingwenz/Alvis/workspace/chenhan/DOGE-NEW/DOGE'

NUM_CMD_PARAMS=$#
if [ $NUM_CMD_PARAMS -ge 3 ]
then
    DOWNSAMPLE=$3
    IMAGE_FOLDER=images_$DOWNSAMPLE
    DEPTH_FOLDER=depths_$DOWNSAMPLE
else
    IMAGE_FOLDER=images
    DEPTH_FOLDER=depths
fi

if ! [ -e $DATASET_PATH/$DEPTH_FOLDER ]
then
    mkdir $DATASET_PATH/$DEPTH_FOLDER
fi

cd /cephyr/users/qingwenz/Alvis/workspace/chenhan/DOGE-NEW/DOGE/Depth-Anything-V2
python -m run --encoder vitl --pred-only --grayscale \
    --img-path $DATASET_PATH/$IMAGE_FOLDER \
    --outdir $DATASET_PATH/$DEPTH_FOLDER

cd $CODE_ROOT_DIR
python -m conerf.utils.make_depth_scale \
    --base_dir $DATASET_PATH \
    --depths_dir $DATASET_PATH/$DEPTH_FOLDER
