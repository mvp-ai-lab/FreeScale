import os
import copy
import warnings
from typing import List

import omegaconf
import numpy as np

from PIL import Image
from conerf.evaluators.camera_sample import CameraSampler
from conerf.utils.utils import setup_seed

warnings.filterwarnings("ignore", category=UserWarning)


def load_images(image_paths: List):
    images = []
    if len(image_paths) == 0:
        return images

    for image_path in image_paths:
        image = Image.open(image_path)
        images.append(image)
    return np.array(images)


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

    # parse YAML config to OmegaConf
    config = load_config(args)
    setup_seed(config.seed)

    scenes = []
    if args.scene != "":  # Overwrite scenes in config file.
        scenes.append(args.scene)
    elif args.scene_list_file != "":
        scenes = read_scenes_from_file(args.scene_list_file)
    else:
        if (
            type(config.dataset.scene) == omegaconf.listconfig.ListConfig
        ):
            for sc in config.dataset.scene:
                scenes.append(sc)
        elif type(config.dataset.scene) == list:
            scenes = config.dataset.scene
        else:
            scenes.append(config.dataset.scene)

    factors = []
    if (
        type(config.dataset.factor) == omegaconf.listconfig.ListConfig # pylint: disable=C0123
    ):
        for factor in config.dataset.factor:
            factors.append(factor)
    elif type(config.dataset.factor) == list: # pylint: disable=C0123
        factors = config.dataset.factor
    else:
        factors.append(config.dataset.factor)

    for i, (scene, factor) in enumerate(zip(scenes, factors)):
        if i < args.start_index:
            continue

        local_config = copy.deepcopy(config)
        local_config.dataset.factor = factor
        local_config.dataset.model_folder = args.model_folder
        local_config.dataset.init_ply_type = args.init_ply_type

        if type(scene) == tuple:
            scene_id, subset = scene[0], scene[1]
            local_config.dataset.scene = scene_id
            local_config.dataset.root_dir = os.path.join(
                local_config.dataset.root_dir, subset)
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
            local_config.dataset.root_dir = os.path.join(
                local_config.dataset.root_dir, args.subset)
            local_config.dataset.scene = scene

        if os.path.exists(os.path.join(local_config.dataset.output_dir, local_config.expname, "renders/freeviews")):
            continue

        sampler = CameraSampler(
            local_config, True, None,
            False, None, False, None
        )
        sampler.eval(split="test")
