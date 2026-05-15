#!/bin/bash
#SBATCH -J lvsm
#SBATCH -N 1 --gpus-per-node=A40:1
#SBATCH -t 3-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=xxx
#SBATCH --output freescale/fv_sampling/%A-%a.out
#SBATCH --error  freescale/fv_sampling/%A-%a.err

module load cuDNN/8.7.0.84-CUDA-11.8.0
cd freescale/lvsm
CONDAENV="[CONDA_ENV_PATH]/bin"

GPU_COUNT=$(nvidia-smi -L | wc -l)
echo "Detected $GPU_COUNT GPU(s). Using all available."

DATASET_PATH="[DATASET_PATH]"
EXP_OUT_DIR="[EXP_OUT_DIR]"
FREEVIEW_DIR="[FREEVIEW_DIR]"
WANDB_API_KEY_PATH="[WANDB_API_KEY_PATH]" # configs/api_keys.yaml

# bz=24 for A40; bz=48 forA100fat
TRAIN_IMG_SIZE=256
$CONDAENV/torchrun --nproc_per_node ${GPU_COUNT} --nnodes 1 --rdzv_id 18635 --rdzv_backend c10d --rdzv_endpoint localhost:29523 \
    train_rayzer.py --config configs/LVSM_scene_decoder_only_fvg.yaml training.train_steps=20000 training.mode=static training.amp_dtype=bf16 model.image_tokenizer.image_size=$TRAIN_IMG_SIZE model.target_pose_tokenizer.image_size=$TRAIN_IMG_SIZE \
    training.dataset_path=${DATASET_PATH} training.dataset_name="data.dataset_scene.DL3DV_VG" \
    training.batch_size_per_gpu=24 training.num_workers=16 training.lr=1e-4 \
    training.wandb.mode=online training.wandb.exp_name=$TRAIN_IMG_SIZE training.wandb.api_key_path=${WANDB_API_KEY_PATH}\
    training.resume_ckpt=data/pretrained_ckpt/scene_decoder_only_$TRAIN_IMG_SIZE.pt training.reset_training_state=True \
    training.checkpoint_dir="$EXP_OUT_DIR/checkpoints/fvgen-$SLURM_JOB_ID-$TRAIN_IMG_SIZE" \
    training.view_selector.min_frame_dist=15 training.view_selector.max_frame_dist=40 \
    training.warm_up=4000 \
    training.view_selector.curriculum_iter=4000 training.view_selector.curriculum_start_min_frame_dist=10 training.view_selector.curriculum_start_max_frame_dist=20 \
    training.num_input_views=4 training.num_target_views=2 \
    training.view_selector.freeview_dir=${FREEVIEW_DIR}
