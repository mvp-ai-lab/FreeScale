# pylint: disable=E0402

import os
import argparse
import json

from typing import List
from pathlib import Path

import numpy as np
from .read_write_model import (
    qvec2rotmat, read_cameras_binary, read_images_binary, parse_colmap_camera_params
)
from .utils import list_images, list_metadata, read_meganerf_mappings, get_filename_from_path


MEGA_NERF_PREPROCESSED_SCENE = ["rubble", "building"]
MEGA_NERF_PREPROCESSED_SCENE_WITH_MAPPINGS = ["Residence", "Campus", "Sci-Art"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="convert colmap to transforms_train/test.json"
    )

    parser.add_argument("--scene_dir", type=str, default="dataset/your_dataset")
    parser.add_argument("--output_dir", type=str, default="dataset/your_dataset")
    parser.add_argument("--scene_name", type=str, default="")
    parser.add_argument("--holdout", type=int, default=50)

    args = parser.parse_args()
    return args


def get_val_images(
    scene_dir: str,
    colmap_image_folder: str = "images",
    scene_name: str = None
) -> List:
    val_dir = os.path.join(scene_dir, "val")

    val_image_names = []

    if scene_name in MEGA_NERF_PREPROCESSED_SCENE_WITH_MAPPINGS:
        metadata_dir = os.path.join(val_dir, "metadata")
        val_metadata_paths = list_metadata(metadata_dir)

        mappings_path = os.path.join(scene_dir, "mappings.txt")
        metadata_to_image_name = read_meganerf_mappings(
            mappings_path=mappings_path
        )[1]

        for metadata_path in val_metadata_paths:
            metadata_name = get_filename_from_path(str(metadata_path))
            image_name = metadata_to_image_name[metadata_name]
            val_image_names.append(f"{colmap_image_folder}/{image_name}")
    else:
        val_image_dir = os.path.join(val_dir, "rgbs")
        assert os.path.exists(val_image_dir), f"{val_image_dir} does not exist!"
        val_image_paths = list_images(val_image_dir)
        for val_image_path in val_image_paths:
            image_name = get_filename_from_path(str(val_image_path))
            val_image_names.append(f"{colmap_image_folder}/{image_name}")

    return val_image_names


def colmap_to_json(scene_dir: str, output_dir: str, holdout: int, scene_name: str = ""):
    """Converts COLMAP's cameras.bin and images.bin to a JSON file.

    Args:
        scene_dir: Path to the reconstruction directory
        output_dir: Path to the output directory.
        camera_model: Camera model used.
        camera_mask_path: Path to the camera mask.
        image_id_to_depth_path: When including sfm-based depth, embed these depth file
                                paths in the exported json
        image_rename_map: Use these image names instead of the names embedded in the COLMAP db

    Returns:
        The number of registered images.
    """

    recon_dir = os.path.join(scene_dir, "sparse", "0")
    cam_id_to_camera = read_cameras_binary(os.path.join(recon_dir, "cameras.bin"))
    im_id_to_image = read_images_binary(os.path.join(recon_dir, "images.bin"))

    frames = []
    for im_id, im_data in im_id_to_image.items():
        # NB: COLMAP uses Eigen / scalar-first quaternions
        # * https://colmap.github.io/format.html
        # * https://github.com/colmap/colmap/blob/bf3e19140f491c3042bfd85b7192ef7d249808ec/
        #   src/base/pose.cc#L75
        # the `rotation_matrix()` handles that format for us.

        rotation = qvec2rotmat(im_data.qvec)

        translation = im_data.tvec.reshape(3, 1)
        w2c = np.concatenate([rotation, translation], 1)
        w2c = np.concatenate([w2c, np.array([[0, 0, 0, 1]])], 0)
        c2w = np.linalg.inv(w2c)
        # Convert from COLMAP's camera coordinate system (OpenCV) to ours (OpenGL)
        c2w[0:3, 1:3] *= -1
        c2w = c2w[np.array([1, 0, 2, 3]), :]
        c2w[2, :] *= -1

        name = im_data.name

        name = Path(f"./images/{name}")

        frame = {
            "file_path": name.as_posix(),
            "transform_matrix": c2w.tolist(),
            "colmap_im_id": im_id,
        }

        frames.append(frame)

    frames.sort(key=lambda x: x["file_path"])

    if set(cam_id_to_camera.keys()) != {1}:
        raise RuntimeError("Only single camera shared for all images is supported.")

    out_train = parse_colmap_camera_params(cam_id_to_camera[1])
    out_test = parse_colmap_camera_params(cam_id_to_camera[1])

    if scene_name not in MEGA_NERF_PREPROCESSED_SCENE:
        frames_train = [f for i, f in enumerate(frames) if i % holdout != 0]
        frames_test = [f for i, f in enumerate(frames) if i % holdout == 0]

        out_train["frames"] = frames_train
        out_test["frames"] = frames_test
    else:
        val_image_names = get_val_images(
            scene_dir=scene_dir,
            scene_name=scene_name,
        )
        frames_train, frames_test = [], []
        for frame in frames:
            if frame["file_path"] not in val_image_names:
                frames_train.append(frame)
            else:
                frames_test.append(frame)

        out_train["frames"] = frames_train
        out_test["frames"] = frames_test

    with open(output_dir / "transforms_train.json", "w", encoding="utf-8") as file:
        json.dump(out_train, file, indent=4)

    with open(output_dir / "transforms_test.json", "w", encoding="utf-8") as file:
        json.dump(out_test, file, indent=4)

    return len(frames)


if __name__ == "__main__":
    init_args = parse_args()

    scene_dir = Path(init_args.scene_dir)
    output_dir = Path(init_args.output_dir)
    holdout = init_args.holdout
    scene_name = init_args.scene_name

    colmap_to_json(scene_dir, output_dir, holdout, scene_name)
