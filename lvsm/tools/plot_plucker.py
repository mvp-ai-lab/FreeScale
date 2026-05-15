"""
# Updated: 2025-05-08 16:32
# 
# Copyright (C) 2023-now, Huawei Technologies Co., Ltd.
# Author: Qingwen Zhang  (https://kin-zhang.github.io/)
#
# Description: Plot a image plucker ray maps.
# 

# Usage:
# python tools/plot_plucker.py --rescale False
# python tools/plot_plucker.py --scene_name 3aaed2e6422d7d57 --metadata_dir /home/qingwen/workspace/LVSM/data/realestate-10k/dataset/lvsm/test/metadata --evaluation_file /home/qingwen/workspace/LVSM/data/evaluation_index_re10k.json 
"""
import fire, json
import numpy as np
import matplotlib.pyplot as plt
import torch
from einops import rearrange
import PIL
from sklearn.decomposition import PCA

def compute_rays(c2w, fxfycxcy, h=None, w=None, device="cpu"):
    """
    Args:
        c2w (torch.tensor): [b, v, 4, 4]
        fxfycxcy (torch.tensor): [b, v, 4]
        h (int): height of the image
        w (int): width of the image
    Returns:
        ray_o (torch.tensor): [b, v, 3, h, w]
        ray_d (torch.tensor): [b, v, 3, h, w]
    """

    b, v = c2w.size()[:2]
    c2w = c2w.reshape(b * v, 4, 4)

    fx, fy, cx, cy = fxfycxcy[:,:, 0], fxfycxcy[:,:,  1], fxfycxcy[:,:,  2], fxfycxcy[:,:,  3]
    h_orig = int(2 * cy.max().item())  # Original height (estimated from the intrinsic matrix)
    w_orig = int(2 * cx.max().item())  # Original width (estimated from the intrinsic matrix)
    if h is None or w is None:
        h, w = h_orig, w_orig

    # in case the ray/image map has different resolution than the original image
    if h_orig != h or w_orig != w:
        fx = fx * w / w_orig
        fy = fy * h / h_orig
        cx = cx * w / w_orig
        cy = cy * h / h_orig

    fxfycxcy = fxfycxcy.reshape(b * v, 4)
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    y, x = y.to(device), x.to(device)
    x = x[None, :, :].expand(b * v, -1, -1).reshape(b * v, -1)
    y = y[None, :, :].expand(b * v, -1, -1).reshape(b * v, -1)
    x = (x + 0.5 - fxfycxcy[:, 2:3]) / fxfycxcy[:, 0:1]
    y = (y + 0.5 - fxfycxcy[:, 3:4]) / fxfycxcy[:, 1:2]
    z = torch.ones_like(x)
    ray_d = torch.stack([x, y, z], dim=2)  # [b*v, h*w, 3]
    ray_d = torch.bmm(ray_d, c2w[:, :3, :3].transpose(1, 2))  # [b*v, h*w, 3]
    ray_d = ray_d / torch.norm(ray_d, dim=2, keepdim=True)  # [b*v, h*w, 3]
    ray_o = c2w[:, :3, 3][:, None, :].expand_as(ray_d)  # [b*v, h*w, 3]

    ray_o = rearrange(ray_o, "(b v) (h w) c -> b v c h w", b=b, v=v, h=h, w=w, c=3)
    ray_d = rearrange(ray_d, "(b v) (h w) c -> b v c h w", b=b, v=v, h=h, w=w, c=3)

    return ray_o, ray_d

def plucker_method(ray_o, ray_d, method="default_plucker"):
    """
    Args:
        images: [b, v, c, h, w]
        ray_o: [b, v, 3, h, w]
        ray_d: [b, v, 3, h, w]
        method: Method for creating pose conditioning
    Returns:
        posed_images: [b, v, c+6, h, w] or [b, v, 6, h, w] if images is None
    """
    if method == "custom_plucker":
        o_dot_d = torch.sum(-ray_o * ray_d, dim=2, keepdim=True)
        nearest_pts = ray_o + o_dot_d * ray_d
        pose_cond = torch.cat([ray_d, nearest_pts], dim=2)
        
    elif method == "aug_plucker":
        o_dot_d = torch.sum(-ray_o * ray_d, dim=2, keepdim=True)
        nearest_pts = ray_o + o_dot_d * ray_d
        o_cross_d = torch.cross(ray_o, ray_d, dim=2)
        pose_cond = torch.cat([o_cross_d, ray_d, nearest_pts], dim=2)
        
    else:  # default_plucker
        o_cross_d = torch.cross(ray_o, ray_d, dim=2)
        pose_cond = torch.cat([o_cross_d, ray_d], dim=2)
    return pose_cond
def rescale01(x):
    """Linearly map x so that min(x)→0 and max(x)→1."""
    x_min, x_max = x.min(), x.max()
    return (x - x_min) / (x_max - x_min + 1e-8)

def main(
    scene_name: str = "hike", # scene name
    metadata_dir: str = "/home/qingwen/workspace/LVSM/data/davis/metadata", # metadata directory
    evaluation_file: str = "/home/qingwen/workspace/LVSM/data/render_dvis.json", # evaluation file
    rescale: bool = True, # whether to rescale the image
):

    print(f"Scene name: {scene_name}, metadata directory: {metadata_dir}, evaluation file: {evaluation_file}")
    # print(f"Output file: {output_file}")
    with open(f"{metadata_dir}/{scene_name}.json", "rb") as f:
        meta_data = json.load(f)
    with open(f"{evaluation_file}", "rb") as f:
        evaluation_data = json.load(f)[scene_name]
    input_dict = [meta_data["frames"][i] for i in evaluation_data['context']]
    target_dict = [meta_data["frames"][i] for i in evaluation_data['target']]
    
    ray_os, ray_ds = {}, {}
    pose_conds = {}
    for mode, data in zip(["input", "target"], [input_dict, target_dict]):
        c2ws = []
        fxfycxcys = []
        for frame in data:
            image = PIL.Image.open(frame["image_path"])
            image_height, image_width = image.size
            fxfycxcy = np.array(frame["fxfycxcy"])
            fxfycxcy = torch.from_numpy(fxfycxcy).float()
            c2w = torch.from_numpy(np.linalg.inv(np.array(frame["w2c"]))).float()
            c2ws.append(c2w)
            fxfycxcys.append(fxfycxcy)
        c2ws = torch.stack(c2ws, dim=0).unsqueeze(0)
        fxfycxcys = torch.stack(fxfycxcys, dim=0).unsqueeze(0)
        ray_o, ray_d = compute_rays(c2ws, fxfycxcys, image_height, image_width)
        pose_cond = plucker_method(ray_o, ray_d, method="default_plucker")
        ray_os[mode] = ray_o.squeeze(0).cpu().numpy()
        ray_ds[mode] = ray_d.squeeze(0).cpu().numpy()
        pose_conds[mode] = pose_cond.squeeze(0).cpu().numpy()
    
    # Reduce to 3 dimensions
    # v, feat_channel, h, w = pose_conds[mode].shape
    # print(f"Pose conditioning shape: {pose_conds[mode].shape}")
    # pca = PCA(n_components=3)

    fig, axs = plt.subplots(2, 3, figsize=(20, 10))
    # axs[0, 0].imshow(ray_os["input"][0].transpose(1, 2, 0))
    for i in range(len(ray_os["input"])):
        input_ray_o = ray_ds["input"][i].transpose(2, 1, 0)
        target_ray_o = ray_ds["target"][i].transpose(2, 1, 0)
        axs[i, 0].imshow(rescale01(input_ray_o) if rescale else input_ray_o)
        axs[i, 0].set_title(f"Input {i} Ray Direction")
        axs[i, 1].imshow(rescale01(target_ray_o) if rescale else target_ray_o)
        axs[i, 1].set_title(f"Target {i} Ray Direction")
        # diff
        diff = input_ray_o - target_ray_o
        # norm
        diff = np.linalg.norm(diff, axis=2)
        im = axs[i, 2].imshow(diff)
        axs[i, 2].set_title(f"(Norm) Diff {i} Ray Direction")
        # add a colorbar for diff
        fig.colorbar(im, ax=axs[i, 2], fraction=0.046, pad=0.04)
    #     # pose cond
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    fire.Fire(main)


