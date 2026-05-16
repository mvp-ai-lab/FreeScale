import copy
import os
import warnings
import logging

import omegaconf
from omegaconf import OmegaConf

from conerf.utils.config import config_parser, load_config
from conerf.utils.utils import setup_seed

from conerf.base.model_base import ModelBase
from conerf.trainers.gaussian_trainer import GaussianSplatTrainer
from conerf.trainers.active_gaussian_trainer import ActiveGaussianSplatTrainer

warnings.filterwarnings("ignore", category=UserWarning)

def create_trainer(
    config: OmegaConf,
    prefetch_dataset=True,
    trainset=None,
    valset=None,
    model: ModelBase = None
):
    """Factory function for training neural network trainers."""
    if config.neural_field_type == "gs":
        if config.trainer.get("active", False):
            trainer = ActiveGaussianSplatTrainer(
                config, prefetch_dataset, trainset, valset, model)
        else:
            trainer = GaussianSplatTrainer(
                config, prefetch_dataset, trainset, valset, model)
    else:
        raise NotImplementedError

    return trainer


def run_cmd(cmd: str):
    os.system(cmd)

    return True


def train(config: OmegaConf):
    trainer = create_trainer(config)
    trainer.update_meta_data()
    trainer.train()
    # print(f"total iteration: {trainer.iteration}")


if __name__ == "__main__":
    args = config_parser()

    logging.basicConfig(
        format='%(asctime)s %(levelname)-6s [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S',
        level=logging.INFO
    )

    # parse YAML config to OmegaConf
    config = load_config(args)
    config["config_file_path"] = args.config

    config.dataset.root_dir = os.path.join(config.dataset.root_dir, args.subset)
    config.dataset.output_dir = os.path.join(config.dataset.output_dir, args.subset)

    assert config.dataset.scene != "" or args.scene != ""

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

        train(local_config)
