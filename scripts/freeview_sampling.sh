#!/bin/bash
#SBATCH -J fv_sampling
#SBATCH -N 1 --gpus-per-node=A40:1
#SBATCH -t 3-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=xxx
#SBATCH --output freescale/fv_sampling/%A-%a.out
#SBATCH --error  freescale/fv_sampling/%A-%a.err

module load GCC/10.3.0 
module load cuDNN/8.7.0.84-CUDA-11.8.0
cd freescale/gaussian_splatting
source miniforge3/etc/profile.d/conda.sh
conda activate freescale

GPU_COUNT=$(nvidia-smi -L | wc -l)
echo "Detected $GPU_COUNT GPU(s). Using all available."

export PYTHONDONTWRITEBYTECODE=1  # Disable python cache.
DATASET_PATH=/data/dl3dv/DL3DV-10K/
CSV_PATH=/data/dl3dv/training_list.txt
OUT_PATH=exps/dl3dv_ff/

# BASE_START_INDEX=7200
# INTERVAL=150
# START_INDEX=$((BASE_START_INDEX + SLURM_ARRAY_TASK_ID * INTERVAL))
# END_INDEX=$((START_INDEX + INTERVAL))

START_INDEX=0
END_INDEX=-1
SUFFIX=3dgs
EXP_OUT_DIR=${OUT_PATH}/out_3dgs
mkdir -p $EXP_OUT_DIR
ACTIVE_EXP_OUT_DIR=${OUT_PATH}/out_active
mkdir -p $ACTIVE_EXP_OUT_DIR
python -m scripts.train.neural_mapping_dispatch $DATASET_PATH \
  --csv_path $CSV_PATH \
  --start_index $START_INDEX \
  --end_index $END_INDEX \
  --suffix $SUFFIX \
  --run_3dgs \
  --exp_out_dir $EXP_OUT_DIR \
  --active_exp_out_dir $ACTIVE_EXP_OUT_DIR \
  --active_training \ # if you want to do per-scene reconstruction w/ freescale

