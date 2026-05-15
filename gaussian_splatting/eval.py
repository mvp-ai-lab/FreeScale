# pylint: disable=[E1101,W0621]

import os
import copy
import warnings
from typing import List

import omegaconf
from omegaconf import OmegaConf

from conerf.evaluators.gaussian_splatting_evaluator import GaussianSplatEvaluator
# from conerf.evaluators.rap_evaluator import RAPEvaluator
from conerf.evaluators.active_evaluator import ActiveEvaluator
from conerf.utils.utils import setup_seed

warnings.filterwarnings("ignore", category=UserWarning)


def create_evaluator(
    config: OmegaConf,
    load_train_data: bool = False,
    trainset=None,
    load_val_data: bool = True,
    valset=None,
    load_test_data: bool = False,
    testset = None,
    models: List = None,
    meta_data: List = None,
    verbose: bool = False,
    device: str = "cuda",
):
    """Factory function for training neural network trainers."""
    if config.trainer.get("active", False):
        evaluator = ActiveEvaluator(
            config, True, trainset,
            load_val_data, valset, load_test_data,
            testset, models, meta_data, verbose, device
            )
    elif config.neural_field_type.find("gs") >= 0:
        evaluator = GaussianSplatEvaluator(
            config, load_train_data, trainset,
            load_val_data, valset, load_test_data,
            testset, models, meta_data, verbose, device
        )
    # elif config.neural_field_type == "localizer":
    #     evaluator = RAPEvaluator(
    #         config, load_train_data, trainset,
    #         load_val_data, valset, load_test_data,
    #         testset, models, meta_data, verbose, device
    #     )
    else:
        raise NotImplementedError

    return evaluator


if __name__ == "__main__":
    from conerf.utils.config import config_parser, load_config
    args = config_parser()

    # parse YAML config to OmegaConf
    config = load_config(args)

    assert config.dataset.scene != "" or args.scene != ""
    config.dataset.root_dir = os.path.join(config.dataset.root_dir, args.subset)

    setup_seed(config.seed)

    if args.val != -1:
        config.dataset.val_interval = args.val

    scenes = []
    if args.scene != "":  # Overwrite scenes in config file.
        scenes.append(args.scene)
    else:
        if (
            type(config.dataset.scene) == omegaconf.listconfig.ListConfig # pylint: disable=C0123
        ):
            scene_list = list(config.dataset.scene)
            for sc in config.dataset.scene:
                scenes.append(sc)
        elif type(config.dataset.scene) == list: # pylint: disable=C0123
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

    for scene, factor in zip(scenes, factors):
        data_dir = os.path.join(config.dataset.root_dir, scene)
        assert os.path.exists(data_dir), f"Dataset does not exist: {data_dir}!"

        local_config = copy.deepcopy(config)
        local_config.expname = (
            f"{config.neural_field_type}_{config.task}_{config.dataset.name}_{scene.split('/')[-1]}"
        )
        local_config.expname = local_config.expname + "_" + args.suffix
        local_config.dataset.scene = scene
        local_config.dataset.factor = factor
        local_config.dataset.model_folder = args.model_folder
        local_config.dataset.init_ply_type = args.init_ply_type
        local_config.dataset.load_specified_images = args.load_specified_images

        evaluator = create_evaluator(
            local_config,
            load_train_data=False,
            trainset=None,
            load_val_data=True,
            valset=None,
            load_test_data=True,
            testset=None,
            verbose=True,
        )
        evaluator.eval(split="val")
        # if local_config.trainer.get("active", False):
        #     evaluator.eval_denoise(split="val", 
        #                            color_correct=local_config.texture.get("color_correct", False)
        #                            )
        #     print("Done")
        # evaluator.eval(split="test")
        # evaluator.export_mesh()
