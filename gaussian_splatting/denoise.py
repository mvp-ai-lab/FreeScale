import os
import copy
import json
import warnings
from typing import List
import numpy as np
from PIL import Image
from conerf.evaluators.camera_sample_rebuttal import CameraSampler
from conerf.utils.utils import setup_seed
warnings.filterwarnings("ignore", category=UserWarning)
import omegaconf
import pandas as pd

def validate_file_count(conf_path: str, target_dir: str) -> bool:
    assert os.path.exists(conf_path), f"No find {conf_path}"
    with open(conf_path, 'r', encoding='utf-8') as f:
        line_count = sum(1 for _ in f)

    with os.scandir(target_dir) as it:
        file_count = sum(1 for entry in it if entry.is_file())

    return line_count == file_count

def read_scenes_from_file(file_path: str):
    scenes = []
    with open(file_path, 'r') as f:
        scene_paths = f.read().splitlines()
    
    for scene_path in scene_paths:
        line = scene_path.strip().split('/')
        scenes.append((line[-1], line[-2]))

    return scenes
   

if __name__ == "__main__":
    from conerf.utils.config import load_config, config_parser
    args = config_parser()
    # root = "/cephyr/users/qingwenz/Alvis/workspace/chenhan/exps/dl3dv_bench/out_active_03"
    config = load_config(args)
    setup_seed(config.seed)

    scenes = []
    if args.scene != "":  # Overwrite scenes in config file.
        scenes.append(args.scene)
    elif args.scene_list_file != "":
        scenes = read_scenes_from_file(args.scene_list_file)
    else:
        if (
            type(config.dataset.scene) == omegaconf.listconfig.ListConfig # pylint: disable=C0123
        ):
            for sc in config.dataset.scene:
                scenes.append(sc)
        elif type(config.dataset.scene) == list: # pylint: disable=C0123
            scenes = config.dataset.scene
        else:
            scenes.append(config.dataset.scene)

    # subset = '4K' 
    # scene_id = '8d7e1e98974898573734cfc6618ae16bd36367d87a551f3818626898eb203513'
    # scene_list = [(scene_id, subset)]

    # scene_list = os.listdir(root)
    # df = pd.read_csv("/cephyr/users/qingwenz/Alvis/workspace/chenhan/data/dl3dv/DL3DV-valid_filter_bench.csv")
    # for i, expname in enumerate(scene_list):
    #     if i < args.start_index:
    #         continue
        
    #     scene_id = expname.split("_")[3]
    #     subset = df.loc[df['scene_id'] == scene_id, 'subset'].tolist()[0]

    for i, scene in enumerate(scenes):
        if i < args.start_index:
            continue
        local_config = copy.deepcopy(config)

        if type(scene) == tuple:
            scene_id, subset = scene[0], scene[1]
            local_config.dataset.scene = scene_id
            local_config.dataset.root_dir = os.path.join(local_config.dataset.root_dir, subset)
            # local_config.dataset.load_from = os.path.join(local_config.dataset.load_from, subset)
            local_config.expname = (
                f"{config.neural_field_type}_{config.task}_{config.dataset.name}_{scene_id}"
            )
            local_config.expname = local_config.expname + "_" + args.suffix
        else:
            local_config.expname = (
                f"{config.neural_field_type}_{config.task}_{config.dataset.name}_{scene.split('/')[-1]}"
            )
            local_config.expname = local_config.expname + "_" + args.suffix
            local_config.dataset.root_dir = os.path.join(local_config.dataset.root_dir, args.subset)
            local_config.dataset.scene = scene
        
        if os.path.exists(os.path.join(local_config.dataset.output_dir, local_config.expname, "renders/freeviews")):
            continue
        
        evaluator = CameraSampler(
            local_config, True, None,
            False, None, False, None
            )
        evaluator.denoise()


       




