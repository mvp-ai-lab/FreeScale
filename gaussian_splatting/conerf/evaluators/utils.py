from typing import Dict, List

import numpy as np

from scripts.preprocess.read_write_model import (
    Camera, BaseImage, Point3D,
    rotmat2qvec, write_model,
)


def to_colmap_camera(
    id: int, width: int, height: int, params: List[float], model: str = "PINHOLE"
):
    return Camera(id=id, model=model, width=width, height=height, params=params)


def to_colmap_cameras(
    intrinsics: np.ndarray, width: int, height: int
) -> Dict:
    cameras = {}
    N = intrinsics.shape[0]
    for i in range(N):
        params = [intrinsics[i, 0, 0], intrinsics[i, 1, 1],
                  intrinsics[i, 0, 2], intrinsics[i, 1, 2]]
        camera = to_colmap_camera(
            i, width=width, height=height, params=params,
        )
        cameras[i] = camera

    return cameras


def to_colmap_images(
    scene_name: str, extrinsics: np.ndarray, image_paths: List[str],
) -> Dict:
    images = {}
    N = extrinsics.shape[0]
    for i in range(N):
        image_path = image_paths[i]
        rotation = extrinsics[i, :3, :3]
        qvec = rotmat2qvec(rotation).astype(np.float32)
        tvec = extrinsics[i, :3, 3].astype(np.float32)
        image_name_start_index = image_path.find(scene_name) + len(scene_name)
        image_name = image_path[image_name_start_index+1:]

        image = BaseImage(
            id=i, qvec=qvec, tvec=tvec, camera_id=i, name=image_name,
            xys=[], point3D_ids=[]
        )
        images[i] = image

    return images


def to_colmap_points(
    points: np.ndarray, rgbs: np.ndarray
) -> Dict:
    points3D = {}
    N = points.shape[0]
    for i in range(N):
        point = Point3D(
            id=i, xyz=points[i], rgb=rgbs[i], error=0,
            image_ids=np.array([]), point2D_idxs=np.array([]),
        )
        points3D[i] = point

    return points3D


def write_colmap_formats(
    seq_name: str,
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    image_paths: List[str],
    points: np.ndarray,
    colors: np.ndarray,
    width: int,
    height: int,
    output_path: str,
    ext: str = '.bin',
):
    colmap_cameras = to_colmap_cameras(intrinsics, width=width, height=height)
    colmap_images = to_colmap_images(
        seq_name, extrinsics=extrinsics, image_paths=image_paths,
    )
    colmap_points3D = to_colmap_points(points, colors)
    write_model(
        colmap_cameras, colmap_images, colmap_points3D, output_path,
        ext=ext
    )
