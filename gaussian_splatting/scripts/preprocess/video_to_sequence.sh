#!/usr/bin/env bash

VIDEO_DIR=$1
INPUT_FILE=$2
OUTPUT_FOLDER=$3
FRAMERATE=$4

mkdir $VIDEO_DIR/$OUTPUT_FOLDER

ffmpeg -i $VIDEO_DIR/$INPUT_FILE.MOV $VIDEO_DIR/$INPUT_FILE.mp4

ffmpeg -i $VIDEO_DIR/$INPUT_FILE.mp4 -pix_fmt yuvj422p \
       -qscale:v 1 -vf fps=$FRAMERATE \
       $VIDEO_DIR/$OUTPUT_FOLDER/img_%06d.png
