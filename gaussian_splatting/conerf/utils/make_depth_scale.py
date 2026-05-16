# pylint: disable=[E1101]
import os
import argparse
import json

import numpy as np
import cv2

from joblib import delayed, Parallel
from scripts.preprocess.read_write_model import *


def get_scales(key, cameras, images, points3d_ordered, args):
    image_meta = images[key]
    cam_intrinsic = cameras[image_meta.camera_id]

    # pts_idx = images[key].point3D_ids
    pts_idx = image_meta.point3D_ids # image_meta[key].point3D_ids

    mask = pts_idx >= 0
    mask *= pts_idx < len(points3d_ordered)

    pts_idx = pts_idx[mask]
    valid_xys = image_meta.xys[mask]

    if len(pts_idx) > 0:
        pts = points3d_ordered[pts_idx]
    else:
        pts = np.array([0, 0, 0])

    R = qvec2rotmat(image_meta.qvec)
    pts = np.dot(pts, R.T) + image_meta.tvec

    inv_colmap_depth_map = 1. / pts[..., 2]
    n_remove = len(image_meta.name.split('.')[-1]) + 1
    inv_mono_depth_map = cv2.imread(
        f"{args.depths_dir}/{image_meta.name[:-n_remove]}.png", cv2.IMREAD_UNCHANGED)

    if inv_mono_depth_map is None:
        return None

    if inv_mono_depth_map.ndim != 2:
        inv_mono_depth_map = inv_mono_depth_map[..., 0]

    inv_mono_depth_map = inv_mono_depth_map.astype(np.float32) / (2**16)
    s = inv_mono_depth_map.shape[0] / cam_intrinsic.height

    maps = (valid_xys * s).astype(np.float32)
    valid = (
        (maps[..., 0] >= 0) *
        (maps[..., 1] >= 0) *
        (maps[..., 0] < cam_intrinsic.width * s) *
        (maps[..., 1] < cam_intrinsic.height * s) * (inv_colmap_depth_map > 0))

    if valid.sum() > 10 and (inv_colmap_depth_map.max() - inv_colmap_depth_map.min()) > 1e-3:
        maps = maps[valid, :]
        inv_colmap_depth_map = inv_colmap_depth_map[valid]
        inv_mono_depth = cv2.remap(
            inv_mono_depth_map, maps[..., 0], maps[..., 1],
            interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)[..., 0]

        # Median / dev
        t_colmap = np.median(inv_colmap_depth_map)
        s_colmap = np.mean(np.abs(inv_colmap_depth_map - t_colmap))

        t_mono = np.median(inv_mono_depth)
        s_mono = np.mean(np.abs(inv_mono_depth - t_mono))
        scale = s_colmap / s_mono
        offset = t_colmap - t_mono * scale
    else:
        scale = 0
        offset = 0
    return {"image_name": image_meta.name[:-n_remove], "scale": scale, "offset": offset}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--base_dir',
        default="../data/big_gaussians/standalone_chunks/campus"
    )
    parser.add_argument(
        '--depths_dir',
        default="../data/big_gaussians/standalone_chunks/campus/depths_any"
    )
    parser.add_argument('--model_type', default="bin")
    args = parser.parse_args()

    cam_intrinsics, images, points3d = read_model(
        os.path.join(args.base_dir, "sparse", "0"), ext=f".{args.model_type}")

    pts_indices = np.array([points3d[key].id for key in points3d])
    pts_xyzs = np.array([points3d[key].xyz for key in points3d])
    points3d_ordered = np.zeros([pts_indices.max()+1, 3])
    points3d_ordered[pts_indices] = pts_xyzs

    depth_param_list = Parallel(n_jobs=-1, backend="threading")(
        delayed(get_scales)(
            key, cam_intrinsics, images, points3d_ordered, args
        ) for key in images
    )

    depth_params = {
        depth_param["image_name"]: {
            "scale": depth_param["scale"], "offset": depth_param["offset"]}
        for depth_param in depth_param_list if depth_param is not None
    }

    with open(f"{args.base_dir}/sparse/0/depth_params.json", "w", encoding='utf-8') as f:
        json.dump(depth_params, f, indent=2)

    # Also save the json file to `manhattan_world' folder.
    os.makedirs(f"{args.base_dir}/sparse/manhattan_world", exist_ok=True)
    with open(
        f"{args.base_dir}/sparse/manhattan_world/depth_params.json", "w", encoding='utf-8'
    ) as f:
        json.dump(depth_params, f, indent=2)
