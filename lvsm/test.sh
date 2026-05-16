export PYTHONPATH="[PROJECT_ROOT]/freescale/lvsm:$PYTHONPATH"

torchrun  --nproc_per_node 1 --nnodes 1 --rdzv_id 18635 --rdzv_backend c10d --rdzv_endpoint localhost:29523 \
    inference.py --config configs/LVSM_scene_decoder_only_fvg.yaml \
    training.dataset_path=data/dl3dv/dl3dv_test.txt \
    training.dataset_name="data.dataset_scene.DL3DV_test" \
    training.num_input_views=4 training.num_target_views=2 \
    training.batch_size_per_gpu=1 \
    training.target_has_input=false \
    training.resume_ckpt=lvsm/data/pretrained_ckpt/scene_decoder_only_$TRAIN_IMG_SIZE.pt \
    training.checkpoint_dir="exps/lvsm/checkpoints/reexp1-5703335-256-4/ckpt_0000000000020000.pt" \
    inference.if_inference=true \
    inference.compute_metrics=true \
    inference.render_video=true \
    inference.view_idx_file_path="freescale/lvsm/data/dl3dv/bench_small_0.json" \
    inference_out_dir=/cephyr/users/qingwenz/Alvis/workspace/chenhan/exps/lvsm/dl3dv_wo_diffusion_small

# torchrun  --nproc_per_node 1 --nnodes 1 --rdzv_id 18635 --rdzv_backend c10d --rdzv_endpoint localhost:29523 \
#     inference.py --config configs/LVSM_scene_decoder_only_fvg.yaml \
#     training.dataset_path=/data/mipnerf360.txt \
#     training.dataset_name="data.dataset_scene.MipNeRFDataset" \
#     training.num_input_views=4 training.num_target_views=2 \
#     training.batch_size_per_gpu=1 \
#     training.target_has_input=false \
#     training.resume_ckpt=lvsm/data/pretrained_ckpt/scene_decoder_only_$TRAIN_IMG_SIZE.pt \
#     training.checkpoint_dir="exps/lvsm/checkpoints/joint-5293872-256-4/ckpt_0000000000020000.pt" \
#     inference.if_inference=true \
#     inference.compute_metrics=true \
#     inference.render_video=false \
#     inference.view_idx_file_path=freescale/lvsm/data/mipnerf360_index_small.json \
#     inference_out_dir=exps/lvsm/mipnerf360_baseline_small

# torchrun  --nproc_per_node 1 --nnodes 1 --rdzv_id 18635 --rdzv_backend c10d --rdzv_endpoint localhost:29523 \
#     inference.py --config configs/LVSM_scene_decoder_only_fvg.yaml \
#     training.dataset_path=/data/tanktemple.txt \
#     training.dataset_name="data.dataset_scene.TTDataset" \
#     training.num_input_views=4 training.num_target_views=2 \
#     training.batch_size_per_gpu=1 \
#     training.target_has_input=false \
#     training.resume_ckpt=lvsm/data/pretrained_ckpt/scene_decoder_only_$TRAIN_IMG_SIZE.pt \
#     training.checkpoint_dir="/cephyr/users/qingwenz/Alvis/workspace/chenhan/exps/lvsm/checkpoints/2KVG2-5312829-256-4/ckpt_0000000000020000.pt" \
#     inference.if_inference=true \
#     inference.compute_metrics=true \
#     inference.render_video=false \
#     inference.view_idx_file_path=freescale/lvsm/data/tanktemples_index_small.json \
#     inference_out_dir=exps/lvsm/tank_temples_small

   