# pylint: disable=[E1101,E1102]

import argparse
import os
import csv
import time
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
from typing import List

import tqdm
import GPUtil
import torch
import numpy as np
import matplotlib.pyplot as plt

from conerf.datasets.utils import minify
from conerf.visualization.pose_visualizer import plot_save_poses
from conerf.pycolmap.pycolmap.scene_manager import SceneManager


PROJECT_ROOT_PATH = "/home/yuchen/Projects/freescale/gaussian_splatting" # Project path
DATASET_NAME = "nerfbusters" # "DL3DV10K"
CONFIG_FILENAME = "nerfbusters" # "custom" # optimization-based
# CONFIG_FILENAME = "gaussian_splatting/custom_ff"


def prepare_args():
    parser = argparse.ArgumentParser(
        description=("Script for rendering novel views of"
                     " synthetic Blender scenes.")
    )
    parser.add_argument(
        "dataset_path", type=str,
        help="Path to the DL3DV10K dataset"
    )
    parser.add_argument(
        "--voc_tree_path", type=str, default="",
        help="Path to the vocabulary tree"
    )
    parser.add_argument(
        "--csv_path", type=str, default="",
        help="Path to the csv file of the DL3DV10K dataset"
    )
    parser.add_argument(
        "--output_path", type=str, default="",
        help="Desired path to the novel view renders."
    )
    parser.add_argument(
        "--suffix", type=str, default="3dgs",
        help="Suffix for a training group."
    )
    parser.add_argument(
        "--downsample", type=int, default=1,
        help="Factors used to downsample images."
    )
    parser.add_argument(
        "--num_gpus", type=int, default=1,
        help="Number of GPUs used to train 3DGS."
    )
    parser.add_argument(
        "--start_index", type=int, default=0,
        help="",
    )
    parser.add_argument(
        "--end_index", type=int, default=-1,
        help="",
    )
    parser.add_argument(
        "--use_manhattan_world", action="store_true",
        help="Whether run reconstruction under the Manhattan World assumption",
    )
    parser.add_argument(
        "--active_training", action="store_true",
        help="Whether run active training for 3DGS",
    )
    parser.add_argument(
        "--exp_out_dir", type=str, default="out",
        help="Experimental output directory for 3DGS training."
    )
    parser.add_argument(
        "--active_exp_out_dir", type=str, default="out_active",
        help="Experimental output directory for active 3DGS training."
    )
    parser.add_argument(
        "--run_sfm", action="store_true",
        help="Whether run Structure-from-Motion",
    )
    parser.add_argument(
        "--run_3dgs", action="store_true",
        help="Whether train 3DGS"
    )
    parser.add_argument(
        "--train_multi_gpus", action="store_true",
        help="Whether train 3DGS"
    )
    parser.add_argument(
        "--subset", default="",
        help="subset of dataset used for DL3DV"
    )

    return parser


def run_cmd(cmd: str):
    os.system(cmd)
    return True


def read_csv_file(csv_file_path: str) -> List:
    scene_names = []

    with open(csv_file_path, newline='', encoding='utf-8') as csv_file:
        reader = csv.reader(csv_file, delimiter=' ')
        for row in reader:
            scene_name = row[0].split(',')
            if csv_file_path.lower().find("dl3dv") >= 0:
                scene_names.append('/'.join([scene_name[1], scene_name[0]]))
            else:
                scene_names.append(scene_name[0])
    scene_names.sort()
    return scene_names


def read_txt_file(txt_file_path: str) -> List:
    scene_paths = []
    with open(txt_file_path, "r", encoding="utf-8") as txt_file:
        for line in txt_file.readlines():
            line = line.strip().split("/")
            scene_path = os.path.join(line[-2], line[-1])
            scene_paths.append(scene_path)
    # scene_paths.sort()
    return scene_paths


def sfm_results_exist(scene_path: str, use_manhattan_world: bool = True):
    assert os.path.exists(scene_path)
    # assert os.path.exists(os.path.join(scene_path, "sparse"))

    if not os.path.exists(os.path.join(scene_path, "sparse")):
        return False

    model_folder = "manhattan_world" if use_manhattan_world else "0"
    sfm_results_path = os.path.join(scene_path, "sparse", model_folder)

    points3d_txt_path = os.path.join(sfm_results_path, "points3D.txt")
    cameras_txt_path = os.path.join(sfm_results_path, "cameras.txt")
    images_txt_path = os.path.join(sfm_results_path, "images.txt")

    all_txt_exists = os.path.exists(points3d_txt_path) and \
        os.path.exists(cameras_txt_path) and os.path.exists(images_txt_path)

    points3d_bin_path = os.path.join(sfm_results_path, "points3D.bin")
    cameras_bin_path = os.path.join(sfm_results_path, "cameras.bin")
    images_bin_path = os.path.join(sfm_results_path, "images.bin")

    all_bin_exists = os.path.exists(points3d_bin_path) and \
        os.path.exists(cameras_bin_path) and os.path.exists(images_bin_path)

    if all_txt_exists or all_bin_exists:
        return True

    return False


def generate_3dgs_training_cmd(
    dataset_path: str,
    scene: str,
    index: int,
    suffix: str = "3dgs",
    gpu_idx: int = 0,
    downsample: int = 1,
    use_manhattan_world: bool = True,
    exp_out_dir: str = None,
    active_training: bool = False,
    active_exp_out_dir: str = None,
):
    if exp_out_dir is None:
        exp_out_dir = os.path.join(dataset_path, "out")
        os.makedirs(exp_out_dir, exist_ok=True)

    scene_exp_out_dir = os.path.join(exp_out_dir, f'gs_nvs_{DATASET_NAME}_{scene.split("/")[-1]}_{suffix}')
    os.makedirs(scene_exp_out_dir, exist_ok=True)

    train_log_path = os.path.join(scene_exp_out_dir, "train_log.txt")
    eval_log_path = os.path.join(scene_exp_out_dir, "eval_log.txt")

    scene_path = os.path.join(dataset_path, scene)  # pylint: disable=W0621
    if downsample > 1:
        minify(scene_path, factors=[downsample])

    cmd = f"export CUDA_VISIBLE_DEVICES={gpu_idx} && "
    train_cmd = f"python train.py --config config/{CONFIG_FILENAME}.yaml " + \
                f"--suffix {suffix} --model_folder sparse " + \
                f"--init_ply_type sparse --scene {scene} " + \
                f"> {train_log_path} 2>&1 && "

    eval_cmd = f"python eval.py --config config/{CONFIG_FILENAME}.yaml " + \
                f"--suffix {suffix} --model_folder sparse " + \
                f"--init_ply_type sparse --scene {scene} " + \
                f"> {eval_log_path} 2>&1 && "
    cmd = cmd + train_cmd + eval_cmd

    # Group 3DGS model files to upload to webui.
    eval_dir = os.path.join(scene_exp_out_dir, "eval/val")
    splats_file = os.path.join(scene_exp_out_dir, "web_splat.splat")
    image_file = os.path.join(eval_dir, "images/rgb_gt/000.png")
    upload_folder = os.path.join(eval_dir, "upload")
    os.makedirs(upload_folder, exist_ok=True)
    new_splats_file = os.path.join(
        upload_folder, f"{DATASET_NAME}_{index}_{scene[-4:]}.splat")
    new_image_file = os.path.join(
        upload_folder, f"{DATASET_NAME}_{index}_{scene[-4:]}.png")
    cmd += f"cp {splats_file} {new_splats_file} && " + \
        f"cp {image_file} {new_image_file}"

    # sample
    cmd += " && "
    if active_exp_out_dir is None:
        active_exp_out_dir = os.path.join(dataset_path, "out_active")
        os.makedirs(active_exp_out_dir, exist_ok=True)
    active_training_config = CONFIG_FILENAME + "_fvg"
    scene_active_exp_out_dir = os.path.join(
            active_exp_out_dir, f'gs_nvs_{DATASET_NAME}_{scene.split("/")[-1]}_{suffix}'
        )
    os.makedirs(scene_active_exp_out_dir, exist_ok=True)
    sample_traj_log_path = os.path.join(scene_active_exp_out_dir, "sample_traj_log.txt")
    sample_cmd = f"cd {PROJECT_ROOT_PATH} && " + \
            f"python -m sample_trajs --config config/{active_training_config}.yaml " + \
            f"--suffix {suffix} --scene {scene} " + \
            f"dataset.load_from={exp_out_dir} > {sample_traj_log_path} 2>&1"
    cmd = cmd + sample_cmd 

    if active_training:
        cmd += " && "
        active_train_log_path = os.path.join(scene_active_exp_out_dir, "active_train_log.txt")
        active_eval_log_path = os.path.join(scene_active_exp_out_dir, "active_eval_log.txt")
        
        active_training_cmd = f"python train.py --config config/{active_training_config}.yaml " + \
                f"--suffix {suffix} --model_folder sparse " + \
                f"--init_ply_type sparse --scene {scene} " + \
                f"> {active_train_log_path} 2>&1 && "
        active_eval_cmd = f"python eval.py --config config/{active_training_config}.yaml " + \
                    f"--suffix {suffix} --model_folder sparse " + \
                    f"--init_ply_type sparse --scene {scene} " + \
                    f"> {active_eval_log_path} 2>&1 && "
        
        resample_cmd = f"cd {PROJECT_ROOT_PATH} && " + \
            f"python -m sample_trajs --config config/{active_training_config}.yaml " + \
            f"--suffix {suffix} --scene {scene} " + \
            f"dataset.load_from={active_exp_out_dir} > {sample_traj_log_path} 2>&1 && "
        cmd = cmd + active_training_cmd + active_eval_cmd + resample_cmd

        # Group 3DGS model files to upload to webui.
        active_eval_dir = os.path.join(scene_active_exp_out_dir, "eval/val")
        active_splats_file = os.path.join(scene_active_exp_out_dir, "web_splat.splat")
        active_image_file = os.path.join(active_eval_dir, "images/rgb_gt/000.png")
        active_upload_folder = os.path.join(active_eval_dir, "upload")
        os.makedirs(active_upload_folder, exist_ok=True)
        new_splats_file = os.path.join(
            active_upload_folder, f"{DATASET_NAME}_{index}_{scene[-4:]}_active.splat")
        new_image_file = os.path.join(
            active_upload_folder, f"{DATASET_NAME}_{index}_{scene[-4:]}_active.png")
        cmd += f"cp {active_splats_file} {new_splats_file} && " + \
            f"cp {active_image_file} {new_image_file}"

    return cmd


def train_multi_gpus(
    dataset_path: str,
    scenes: List,
    start_index: int,
    end_index: int,
    suffix: str = "3dgs",
    num_gpus: int = 1,
    downsample: int = 1,
    use_manhattan_world: bool = True,
    exp_out_dir: str = None,
    active_training: bool = False,
    active_exp_out_dir: str = None,
    excluded_gpus: List = [],
    subset: str = ""
):
    queue = Queue()
    pool = ThreadPoolExecutor(max_workers=num_gpus)
    reserved_gpus = set()
    future_to_job = {}

    # exp_out_dir = os.path.join(dataset_path, "out")

    # Add items to the queue.
    for i, scene in enumerate(scenes):  # pylint: disable=W0621
        if i < start_index or i > end_index:
            continue
        queue.put((i, scene))

    while not queue.empty() or future_to_job:
        # Get the list of available GPUs, not including those that are reserved.
        all_available_gpus = set(GPUtil.getAvailable(
            order="first", limit=10, maxMemory=0.5,
            excludeID=excluded_gpus
        ))
        available_gpus = list(all_available_gpus - reserved_gpus)

        # Launch new jobs on available GPUs.
        while available_gpus and not queue.empty():
            gpu = available_gpus.pop(0)
            i, scene = queue.get()
            print(f'Submit job #{i} to GPU #{gpu}')
            CMD = generate_3dgs_training_cmd(
                dataset_path, scene, i, suffix, gpu, downsample, use_manhattan_world,
                exp_out_dir, active_training, active_exp_out_dir
            )
            future = pool.submit(run_cmd, CMD)
            future_to_job[future] = (gpu, i)

            reserved_gpus.add(gpu)

        # Check for completed jobs and remove them from the list of running jobs
        # and release the GPUs they were using.
        done_futures = [future for future in future_to_job if future.done()]
        for future in done_futures:
            job = future_to_job.pop(future)
            gpu = job[0]
            reserved_gpus.discard(gpu)
            print(f"Scene #{job[1]} has finished, releasing GPU #{gpu}")

        # (Optional) You may want to introduce a small delay here to prevent this loop from
        # spinning very fast when there are no GPUs available.
        time.sleep(5)


def visualize_camera_poses(data_dir: str, cam_depth: float = 0.2, axis_len: float = 3.0):
    colmap_dir = os.path.join(data_dir, "sparse/manhattan_world")
    assert os.path.exists(colmap_dir), \
        f"colmap model path {colmap_dir} does not exist!"

    manager = SceneManager(colmap_dir, load_points=True)
    manager.load()

    # points3d = torch.from_numpy(manager.points3D).float()
    image_data = manager.images
    bottom = np.array([0, 0, 0, 1]).reshape(1, 4)
    w2c_mats = []

    for k in image_data:
        im_data = image_data[k]

        world_to_cam = np.concatenate([
            np.concatenate(
                [im_data.R(), im_data.tvec.reshape(3, 1)], 1), bottom
        ], axis=0)
        w2c_mats.append(world_to_cam)
    w2c_mats = np.stack(w2c_mats, axis=0)
    camtoworlds = torch.linalg.inv(torch.from_numpy(w2c_mats)).float()

    plt.clf()
    fig = plt.figure(figsize=(16, 8))
    plot_save_poses(
        cam_depth, fig,
        pose=camtoworlds,
        path=data_dir,
        ep='pose',
        axis_len=axis_len,
    )


if __name__ == "__main__":
    parser = prepare_args()
    args = parser.parse_args()

    dataset_path = args.dataset_path
    assert os.path.exists(dataset_path), \
        f"Invalid dataset path: {dataset_path}"
    
    is_csv = args.csv_path.find('.csv') >= 0
    if is_csv:
        scene_names = read_csv_file(args.csv_path)
    else:
        assert args.csv_path.find(".txt") >= 0, \
            "File must be a '.txt' file or a '.csv' file!"
        scene_names = read_txt_file(args.csv_path)

    # Check validity of the vocabulary tree path.
    if args.run_sfm:
        assert os.path.exists(args.voc_tree_path), \
            f"Vocabulary tree {args.voc_tree_path} does not exist!"

    # Filter non-exist scenes.
    valid_scene_names = []
    for i, scene in enumerate(scene_names):
        scene_path = os.path.join(dataset_path, scene)
        if os.path.exists(scene_path):
            valid_scene_names.append(scene)
        else:
            print(scene_path)

    pbar = tqdm.trange(len(valid_scene_names), desc="Checking SfM results...")
    num_scenes_to_reconstruct = 0
    for i, scene in enumerate(valid_scene_names):
        scene_path = os.path.join(dataset_path, scene)
        if not sfm_results_exist(scene_path, args.use_manhattan_world):
            num_scenes_to_reconstruct += 1
            print(f'scene {scene} has not been reconstructed!')
        pbar.update(1)

    print(f'Total Scenes: {len(scene_names)}')
    print(f'Total Valid Scenes: {len(valid_scene_names)}')
    print(f'Number of Scenes to Reconstruct: {num_scenes_to_reconstruct}')

    # Check index validity.
    assert args.end_index < len(valid_scene_names), "invalid ending index!"
    if args.end_index == -1:
        args.end_index = len(valid_scene_names) - 1

    # NOTE: If you want to train 3DGS on a single machine with multiple GPUs, try:
    if args.train_multi_gpus:
        train_multi_gpus(
            dataset_path, valid_scene_names,
            args.start_index, args.end_index,
            args.suffix, args.num_gpus,
            args.downsample, args.use_manhattan_world,
            args.exp_out_dir, args.active_training,
            args.active_exp_out_dir,
            subset=args.subset
        )
    else:
        # exp_out_dir = os.path.join(dataset_path, "out")

        pbar = tqdm.trange(args.end_index - args.start_index + 1, desc="Neural Mapping...")
        for i, scene in enumerate(valid_scene_names):
            # Skip scenes out of the region.
            if i < args.start_index or i > args.end_index:
                continue

            scene_path = os.path.join(dataset_path, scene)
            if not os.path.exists(scene_path):
                continue
            # assert os.path.exists(scene_path), f"scene {scene_path} does not exist!"

            if args.run_sfm:
                if sfm_results_exist(scene_path, args.use_manhattan_world):
                    print(f'Scene {scene_path} already reconstructed!')
                    continue

                print(f'Running SfM for scene {scene_path}')
                RUN_SFM_CMD = f"cd {PROJECT_ROOT_PATH}/scripts/preprocess && " + \
                    f"./colmap_mapping.sh {scene_path} {scene_path} {args.voc_tree_path} 100 0"
                os.system(RUN_SFM_CMD)
                # Visualize colmap poses to images under the folder.
                visualize_camera_poses(scene_path)

            if args.run_3dgs:
                print('Training 3DGS...')
                RUN_3DGS_CMD = generate_3dgs_training_cmd(
                    dataset_path, scene, i, args.suffix,
                    downsample=args.downsample,
                    use_manhattan_world=args.use_manhattan_world,
                    exp_out_dir=args.exp_out_dir,
                    active_training=args.active_training,
                    active_exp_out_dir=args.active_exp_out_dir,
                    # subset=args.subset
                )
                os.system(RUN_3DGS_CMD)

                # TODO(chenyu): add reconstructed splats file to database for visualization.

            pbar.update(1)
