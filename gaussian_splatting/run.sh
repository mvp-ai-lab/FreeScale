export PYTHONDONTWRITEBYTECODE=1  # Disable python cache.
DATASET_PATH=/cephyr/users/qingwenz/Alvis/data/dl3dv/DL3DV-10K/
CSV_PATH=data/scene_list.txt
OUT_PATH=exps/reconstruction
START_INDEX=0
END_INDEX=-1
SUFFIX=3dgs
EXP_OUT_DIR=/cephyr/users/qingwenz/Alvis/workspace/chenhan/exps/dl3dv_bench/out_3dgs_03
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
  --active_training \



sample_cmd = f"python sample_trajs.py --config config/gaussian_splatting/custom_sample.yaml " + \
             f"--scene_list_file /cephyr/users/qingwenz/Alvis/workspace/chenhan/dl3dv_1105.txt " + \
             f"--suffix 3dgs --start_index 0 "

eval_cmd = f"python eval.py --config config/gaussian_splatting/custom_fvg.yaml " + \
            f"--suffix {suffix} --model_folder sparse " + \
            f"--init_ply_type sparse --scene {scene_id} "