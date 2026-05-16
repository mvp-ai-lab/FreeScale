# pylint: disable=[E1101,C0103]

import copy
import math
from typing import Dict, List, Tuple
from io import BytesIO
from functools import reduce
from operator import mul

import torch
import torch.nn as nn
import numpy as np
import tqdm

from plyfile import PlyData, PlyElement
# from simple_knn._C import distCUDA2
from sklearn.neighbors import NearestNeighbors
from gsplat.compression import PngCompression

from conerf.datasets.utils import BasicPointCloud
# from conerf.geometry.camera import Camera
from conerf.gaussian_fields.utils import (
    quaternion_to_rotation_mat,
    rotation_mat_left_multiply_scale_mat,
    strip_symmetric
)
from conerf.gaussian_fields.sh_utils import RGB2SH, eval_sh

def knn(x: torch.Tensor, K: int = 4) -> torch.Tensor:
    x_np = x.cpu().numpy()
    model = NearestNeighbors(n_neighbors=K, metric="euclidean").fit(x_np)
    distances, _ = model.kneighbors(x_np)
    return torch.from_numpy(distances).to(x)


def detach_tensor_to_numpy(tensor: torch.Tensor):
    return tensor.detach().cpu().numpy()


def inverse_sigmoid(x: torch.Tensor):
    """
    The inverse of the sigmoid function.
    """
    return torch.log(x / (1 - x))


def replace_tensor_to_optimizer(tensor, optimizer, name):
    optimizable_tensors = {}
    for group in optimizer.param_groups:
        if group["name"] == name:
            stored_state = optimizer.state.get(group['params'][0], None)
            stored_state["exp_avg"] = torch.zeros_like(tensor)
            stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

            del optimizer.state[group['params'][0]]
            group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
            optimizer.state[group['params'][0]] = stored_state

            optimizable_tensors[group["name"]] = group["params"][0]

    return optimizable_tensors


def cat_tensors_to_optimizer(tensors_dict: Dict, optimizer):
    optimizable_tensors = {}
    for group in optimizer.param_groups:
        if 'mlp' in group['name'] or 'conv' in group['name'] or \
           'feat_base' in group['name'] or 'offset_model' in group['name']:
            continue

        assert len(group["params"]) == 1
        extension_tensor = tensors_dict[group["name"]]
        stored_state = optimizer.state.get(group['params'][0], None)
        if stored_state is not None:
            stored_state["exp_avg"] = torch.cat((
                stored_state["exp_avg"], torch.zeros_like(extension_tensor)
            ), dim=0)
            stored_state["exp_avg_sq"] = torch.cat((
                stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)
            ), dim=0)

            del optimizer.state[group['params'][0]]
            group["params"][0] = nn.Parameter(torch.cat(
                (group["params"][0], extension_tensor), dim=0
            ).requires_grad_(True))
            optimizer.state[group['params'][0]] = stored_state

            optimizable_tensors[group["name"]] = group["params"][0]
        else:
            group["params"][0] = nn.Parameter(torch.cat(
                (group["params"][0], extension_tensor), dim=0
            ).requires_grad_(True))
            optimizable_tensors[group["name"]] = group["params"][0]

    return optimizable_tensors


def prune_optimizer(mask, optimizer):
    optimizable_tensors = {}
    for group in optimizer.param_groups:
        if 'mlp' in group['name'] or 'conv' in group['name'] or 'offset_model' in group['name']:
            continue

        stored_state = optimizer.state.get(group['params'][0], None)
        if stored_state is not None:
            stored_state["exp_avg"] = stored_state["exp_avg"][mask]
            stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

            del optimizer.state[group['params'][0]]
            group["params"][0] = nn.Parameter(
                (group["params"][0][mask].requires_grad_(True)))
            optimizer.state[group['params'][0]] = stored_state

            optimizable_tensors[group["name"]] = group["params"][0]
        else:
            group["params"][0] = nn.Parameter(
                group["params"][0][mask].requires_grad_(True))
            optimizable_tensors[group["name"]] = group["params"][0]

    return optimizable_tensors


def build_covariance_from_scaling_rotation(scaling, scaling_modifier, quaternion):
    L = rotation_mat_left_multiply_scale_mat(
        scaling_modifier * scaling, quaternion)
    actual_covariance = L @ L.transpose(1, 2)
    symm = strip_symmetric(actual_covariance)

    return symm


def _get_fourier_features(xyz: torch.Tensor, num_features=3):
    xyz = torch.from_numpy(xyz).contiguous().to(dtype=torch.float32)
    xyz = xyz - xyz.mean(dim=0, keepdim=True)
    xyz = xyz / torch.quantile(xyz.abs(), 0.97, dim=0) * 0.5 + 0.5
    freqs = torch.repeat_interleave(
        2**torch.linspace(0, num_features-1, num_features, dtype=xyz.dtype, device=xyz.device), 2)
    offsets = torch.tensor([0, 0.5 * math.pi] * num_features,
                           dtype=xyz.dtype, device=xyz.device)
    feat = xyz[..., None] * freqs[None, None] * \
        2 * math.pi + offsets[None, None]
    feat = torch.sin(feat).view(-1, reduce(mul, feat.shape[1:]))
    return feat


class GaussianSplatModel:
    def __init__(
        self,
        max_sh_degree: int = 3,
        percent_dense: float = 0.01,
        app_feat_dim: int = None,
        device: str = "cuda"
    ) -> None:
        self.device = device
        self.active_sh_degree = 0
        self.max_sh_degree = max_sh_degree
        self.percent_dense = percent_dense

        self._splats = {}
        self.app_feat_dim = app_feat_dim

        self._setup()

    def _setup(self):
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.quaternion_activation = torch.nn.functional.normalize

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        if self.app_feat_dim is None:
            for i in range(self._splats["sh0"].shape[1] * self._splats["sh0"].shape[2]):
                l.append(f'f_dc_{i}')
            for i in range(self._splats["shN"].shape[1] * self._splats["shN"].shape[2]):
                l.append(f'f_rest_{i}')
        else:
            for i in range(self._splats["colors"].shape[1]):
                l.append(f'color_{i}')
            for i in range(self._splats["features"].shape[1]):
                l.append(f'feature_{i}')
        l.append('opacity')
        for i in range(self._splats["scales"].shape[1]):
            l.append(f'scale_{i}')
        for i in range(self._splats["quats"].shape[1]):
            l.append(f'rot_{i}')
        return l

    @property
    def get_raw_quaternion(self):
        """
        Return the original quaternions.
        """
        return self._splats["quats"]

    def set_raw_quaternion(self, quaternion: torch.Tensor):
        """
        Setter for quaternions.
        """
        self._splats["quats"] = quaternion

    def set_opt_raw_quaternion(self, quaternion: torch.Tensor):
        """
        Setter for quaternions.
        """
        self._splats["quats"] = nn.Parameter(quaternion.requires_grad_(True))

    @property
    def get_quaternion(self):
        """
        Return the normalized quaternions.
        """
        return self.quaternion_activation(self._splats["quats"])

    @property
    def get_raw_scaling(self):
        """
        Return the original scaling matrix.
        """
        return self._splats["scales"]

    def set_raw_scaling(self, scaling: torch.Tensor):
        """
        Setter for scaling matrix.
        """
        self._splats["scales"] = scaling

    def set_opt_raw_scaling(self, scaling: torch.Tensor):
        self._splats["scales"] = nn.Parameter(scaling.requires_grad_(True))

    @property
    def get_scaling(self):
        """
        Return the scaling matrix after applying activation function.
        """
        return self.scaling_activation(self._splats["scales"])

    @property
    def get_xyz(self):
        return self._splats["means"]

    def set_xyz(self, xyz: torch.Tensor):
        self._splats["means"] = xyz

    def set_opt_xyz(self, xyz: torch.Tensor):
        self._splats["means"] = nn.Parameter(xyz.requires_grad_(True))

    @property
    def get_features_dc(self):
        return self._splats["sh0"] if self.app_feat_dim is None else \
               self._splats["features"]

    def set_features_dc(self, features_dc: torch.Tensor):
        if self.app_feat_dim is None:
            self._splats["sh0"] = features_dc
        else:
            self._splats["features"] = features_dc

    def set_opt_features_dc(self, features_dc: torch.Tensor):
        if self.app_feat_dim is None:
            self._splats["sh0"] = nn.Parameter(features_dc.requires_grad_(True))
        else:
            self._splats["features"] = nn.Parameter(features_dc.requires_grad_(True))

    @property
    def get_features_rest(self):
        return self._splats["shN"]

    def set_features_rest(self, features_rest: torch.Tensor):
        self._splats["shN"] = features_rest

    def set_opt_features_rest(self, features_rest: torch.Tensor):
        self._splats["shN"] = nn.Parameter(features_rest.requires_grad_(True))

    @property
    def get_features(self):
        features = self.get_features_dc
        if self.app_feat_dim is None:
            features_rest = self.get_features_rest
            features = torch.cat((features, features_rest), dim=1)
        return features

    @property
    def get_raw_opacity(self):
        """
        Return the original opacity.
        """
        return self._splats["opacities"]

    def set_raw_opacity(self, opacity: torch.Tensor):
        self._splats["opacities"] = opacity

    def set_opt_raw_opacity(self, opacity: torch.Tensor):
        self._splats["opacities"] = nn.Parameter(opacity.requires_grad_(True))

    @property
    def get_opacity(self):
        """
        Return the opacity after applying activation function.
        """
        return self.opacity_activation(self._splats["opacities"])

    def get_covariance(self, scaling_modifier: float = 1.0):
        return self.covariance_activation(
            self.get_scaling,
            scaling_modifier,
            self.get_raw_quaternion,
        )

    def set_colors(self, colors: torch.Tensor):
        self._splats["colors"] = colors

    def set_opt_colors(self, colors: torch.Tensor):
        self._splats["colors"] = nn.Parameter(colors.requires_grad_(True))

    @property
    def get_colors(self):
        return self._splats["colors"]

    def get_all_properties(self, indices: torch.Tensor = None) -> Tuple:
        if indices is None:
            return (
                self._splats['means'], self._splats["sh0"], self._splats["shN"],
                self._splats["scales"], self._splats["quats"], self._splats["opacities"]
            )
        return (
            self._splats['means'][indices],
            self._splats["sh0"][indices],
            self._splats["shN"][indices],
            self._splats["scales"][indices],
            self._splats["quats"][indices],
            self._splats["opacities"][indices]
        )

    def get_param_dict(self) -> Dict[str, torch.nn.Parameter]:
        return self._splats

    @torch.no_grad()
    def compress(self, compress_dir: str):
        compress_method = PngCompression()
        compress_method.compress(compress_dir, copy.deepcopy(self._splats))

    @torch.no_grad()
    def decompress(self, compress_dir: str):
        compress_method = PngCompression()
        self._splats = compress_method.decompress(compress_dir)

    def get_sub_gaussians(self, indices: torch.Tensor):
        sub_gaussians = GaussianSplatModel(
            self.max_sh_degree, self.percent_dense,
        )
        sub_gaussians.active_sh_degree = self.active_sh_degree
        sub_gaussians.set_opt_xyz(self._xyz[indices, :])
        sub_gaussians.set_opt_features_dc(self._features_dc[indices, :])
        sub_gaussians.set_opt_features_rest(self._features_rest[indices, :])
        sub_gaussians.set_opt_raw_scaling(self._scaling[indices, :])
        sub_gaussians.set_opt_raw_quaternion(self._quaternion[indices, :])
        sub_gaussians.set_opt_raw_opacity(self._opacity[indices, :])

        return sub_gaussians

    def extract_sub_gaussians(self, indices=None):
        self._xyz = self._xyz[indices]
        self._features_dc = self._features_dc[indices]
        self._features_rest = self._features_rest[indices]
        self._scaling = self._scaling[indices]
        self._quaternion = self._quaternion[indices]
        self._opacity = self._opacity[indices]

    def reinitialize(self):
        self._xyz = torch.zeros_like(self._xyz)
        self._features_dc = torch.zeros_like(self._features_dc)
        self._features_rest = torch.zeros_like(self._features_rest)
        self._scaling = torch.zeros_like(self._scaling)
        self._quaternion = torch.zeros_like(self._quaternion)
        self._opacity = torch.zeros_like(self._opacity)

    @torch.no_grad()
    def plus_gaussians(self, gaussians, indices: torch.Tensor):
        self._xyz[indices, :] += gaussians.get_xyz
        self._features_dc[indices, :] += gaussians.get_features_dc
        self._features_rest[indices, :] += gaussians.get_features_rest
        self._scaling[indices, :] += gaussians.get_raw_scaling
        self._quaternion[indices, :] += gaussians.get_raw_quaternion
        self._opacity[indices, :] += gaussians.get_raw_opacity

    @torch.no_grad()
    def average_gaussians(self, count: torch.Tensor):
        self._xyz /= count.expand(-1, 3)
        self._features_dc /= count.unsqueeze(-1).expand(-1,
                                                        self._features_dc.shape[-2], 3)
        self._features_rest /= count.unsqueeze(-1).expand(-1,
                                                          self._features_rest.shape[-2], 3)
        self._scaling /= count.expand(-1, 3)
        self._quaternion /= count.expand(-1, 4)
        self._opacity /= count

    def eval(self):
        is_training = self._splats["means"].is_leaf
        if is_training:
            self._splats["means"].requires_grad = False
            self._splats["scales"].requires_grad = False
            self._splats["quats"].requires_grad = False
            self._splats["opacities"].requires_grad = False
            if self.app_feat_dim is None:
                self._splats["sh0"].requires_grad = False
                self._splats["shN"].requires_grad = False
            else:
                self._splats["features"].requires_grad = False
                self._splats["colors"].requires_grad = False

    def train(self):
        is_training = self._splats["means"].is_leaf
        if is_training:
            self._splats["means"].requires_grad = True
            self._splats["scales"].requires_grad = True
            self._splats["quats"].requires_grad = True
            self._splats["opacities"].requires_grad = True
            if self.app_feat_dim is None:
                self._splats["sh0"].requires_grad = True
                self._splats["shN"].requires_grad = True
            else:
                self._splats["features"].requires_grad = True
                self._splats["colors"].requires_grad = True

    def increase_SH_degree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def reset_opacity(self, optimizer):
        opacities_new = inverse_sigmoid(torch.min(
            self.get_opacity, torch.ones_like(self.get_opacity) * 0.01))
        optimizable_tensors = replace_tensor_to_optimizer(
            opacities_new, optimizer, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def densification_postfix(
        self, optimizer, new_xyz, new_opacities, new_scaling, new_quaternion,
        new_features_dc, new_features_rest=None, new_colors=None,
    ):
        tensors_dict = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            # "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "quaternion": new_quaternion,
        }
        if new_features_rest is not None:
            tensors_dict["f_rest"] = new_features_rest
        if new_colors is not None:
            tensors_dict["color"] = new_colors

        device = self._xyz.device
        optimizable_tensors = cat_tensors_to_optimizer(tensors_dict, optimizer)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        # self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._quaternion = optimizable_tensors["quaternion"]
        if 'f_rest' in optimizable_tensors.keys():  # pylint: disable=[C0201]
            self._features_rest = optimizable_tensors["f_rest"]
        if 'color' in optimizable_tensors.keys():  # pylint: disable=[C0201]
            self._colors = optimizable_tensors["color"]

        self.xyz_gradient_accum = torch.zeros(
            (self.get_xyz.shape[0], 1), device=device)
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=device)
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=device)

    def prune_points(self, mask, optimizer):
        valid_points_mask = ~mask
        optimizable_tensors = prune_optimizer(valid_points_mask, optimizer)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        # self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._quaternion = optimizable_tensors["quaternion"]
        if 'f_rest' in optimizable_tensors.keys():  # pylint: disable=[C0201]
            self._features_rest = optimizable_tensors["f_rest"]
        if 'color' in optimizable_tensors.keys():  # pylint: disable=[C0201]
            self._colors = optimizable_tensors["color"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    @torch.no_grad()
    def prune_gaussians_with_opt(self, percent: float, import_score: List, optimizer):
        sorted_tensor, _ = torch.sort(import_score, dim=0)
        index_nth_percentile = int(percent * (sorted_tensor.shape[0] - 1))
        value_nth_percentile = sorted_tensor[index_nth_percentile]
        prune_mask = (import_score <= value_nth_percentile).squeeze()
        self.prune_points(prune_mask, optimizer)

    @torch.no_grad()
    def prune_gaussians(self, percent: float, import_score: List):
        sorted_tensor, _ = torch.sort(import_score, dim=0)
        index_nth_percentile = int(percent * (sorted_tensor.shape[0] - 1))
        value_nth_percentile = sorted_tensor[index_nth_percentile]
        prune_mask = (import_score <= value_nth_percentile).squeeze()

        valid_mask = ~prune_mask
        self._xyz = self._xyz[valid_mask]
        self._features_dc = self._features_dc[valid_mask]
        self._features_rest = self._features_rest[valid_mask]
        self._opacity = self._opacity[valid_mask]
        self._scaling = self._scaling[valid_mask]
        self._quaternion = self._quaternion[valid_mask]

    def densify_and_clone(self, grads, grad_threshold, scene_extent, optimizer):
        # Extract points that satisfy the gradient condition.
        selected_pts_mask = torch.where(torch.norm(
            grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling,
                      dim=1).values <= self.percent_dense * scene_extent
        )

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask] \
            if self.app_feat_dim is None else None
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_quaternion = self._quaternion[selected_pts_mask]
        new_colors = self._colors[selected_pts_mask] \
            if self.app_feat_dim is not None else None

        self.densification_postfix(
            optimizer, new_xyz, new_opacities, new_scaling, new_quaternion,
            new_features_dc, new_features_rest, new_colors,
        )

    def densify_and_split(
        self, grads, grad_threshold, scene_extent, optimizer, num_replica: int = 2
    ):
        n_init_points = self.get_xyz.shape[0]
        device = self.get_xyz.device

        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device=device)
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(
            padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling,
                      dim=1).values > self.percent_dense * scene_extent
        )

        stds = self.get_scaling[selected_pts_mask].repeat(num_replica, 1)
        means = torch.zeros((stds.size(0), 3), device=device)
        samples = torch.normal(mean=means, std=stds)
        rotations = quaternion_to_rotation_mat(
            self._quaternion[selected_pts_mask]
        ).repeat(num_replica, 1, 1)
        new_xyz = torch.bmm(
            rotations, samples.unsqueeze(-1)
        ).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(num_replica, 1)
        new_scaling = self.scaling_inverse_activation(
            self.get_scaling[selected_pts_mask].repeat(
                num_replica, 1) / (0.8 * num_replica)
        )
        new_quaternion = self._quaternion[selected_pts_mask].repeat(
            num_replica, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(num_replica, 1)
        if self.app_feat_dim is None:
            new_features_dc = self._features_dc[selected_pts_mask].repeat(
                num_replica, 1, 1)
            new_features_rest = self._features_rest[selected_pts_mask].repeat(
                num_replica, 1, 1)
            new_colors = None
        else:
            new_features_dc = self._features_dc[selected_pts_mask].repeat(
                num_replica, 1)
            new_features_rest = None
            new_colors = self._colors[selected_pts_mask].repeat(num_replica, 1)

        self.densification_postfix(
            optimizer, new_xyz, new_opacity, new_scaling, new_quaternion,
            new_features_dc, new_features_rest, new_colors,
        )

        prune_filter = torch.cat((
            selected_pts_mask,
            torch.zeros(num_replica * selected_pts_mask.sum(),
                        device=device, dtype=bool)
        ))
        self.prune_points(prune_filter, optimizer)

    def densify_and_prune(
        self,
        max_grad,
        min_opacity,
        extent,
        max_screen_size,
        optimizer,
        bounding_box: torch.Tensor = None,
        # parameters for motion deblur.
        prune_depth: bool = False,
        tar_range: int = 3,
    ):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent, optimizer)
        self.densify_and_split(grads, max_grad, extent, optimizer)

        if prune_depth:
            depth = self.get_xyz[..., -1]
            min_depth, max_depth = depth.amin(), depth.amax()
            norm_depth = (depth - min_depth) / (max_depth -
                                                min_depth) * (tar_range - 1) + 1
            min_opacity = min_opacity / norm_depth
            min_opacity = min_opacity[..., None]

        prune_mask = (self.get_opacity < min_opacity).squeeze()

        if bounding_box is not None:
            invalid_pos_mask = (self.get_xyz[:, 2] < bounding_box[2]).squeeze()
            prune_mask = torch.logical_or(prune_mask, invalid_pos_mask)

        if max_screen_size is not None:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(
                torch.logical_or(prune_mask, big_points_vs),
                big_points_ws
            )
        self.prune_points(prune_mask, optimizer)

        torch.cuda.empty_cache()

    def add_densification_stats(self, screen_space_points, update_filter, width, height):
        if type(screen_space_points) == list:
            for i in range(len(screen_space_points)):  # pylint: disable=C0200
                grad = screen_space_points[i].grad.squeeze(0)  # [N,2]
                # Normalize the gradient to [-1,1] screen size.
                grad[:, 0] *= width * 0.5
                grad[:, 1] *= height * 0.5
                self.xyz_gradient_accum[update_filter[i]] += torch.norm(
                    grad[update_filter[i], :2],
                    dim=-1,
                    keepdim=True,
                )
                self.denom[update_filter[i]] += 1 / len(update_filter)
        else:
            participated_pixels = 1
            grad = screen_space_points.grad.squeeze(0)  # [N,2]
            # Normalize the gradient to [-1,1] screen size.
            grad[:, 0] *= width * 0.5
            grad[:, 1] *= height * 0.5
            self.xyz_gradient_accum[update_filter] += torch.norm(
                grad[update_filter, :2], dim=-1, keepdim=True
            ) * participated_pixels
            self.denom[update_filter] += participated_pixels

    def allocate_extra_points(
        self,
        distance: float = 10000.0,
        num_nearest_neighbor: int = 4,
        num_points: int = 100000,
        bound: int = 50,
    ):
        existing_points = self.get_xyz
        existing_color = self.get_features.transpose(
            1, 2).view(-1, 3, (self.max_sh_degree + 1) ** 2)
        min_dist = torch.tensor([distance], device=self.device)
        sorted_existing_points = existing_points.sort(0)[0]
        bbox_min = sorted_existing_points[bound]
        bbox_max = sorted_existing_points[-bound]

        extra_points = torch.rand((num_points, 3), device=self.device)
        extra_points = extra_points * (bbox_max - bbox_min) + bbox_min
        dummy_color = torch.rand(
            (3, (self.max_sh_degree + 1) ** 2), device=self.device)
        mask_points = torch.ones(
            num_points, dtype=torch.bool, device=self.device)

        def find_nearest_neighbors(new_point):
            distances = torch.norm(existing_points - new_point, dim=1)
            nearest_indices = torch.topk(-distances,
                                         num_nearest_neighbor).indices
            return nearest_indices, distances

        interpolated_colors = []
        for i, new_point in enumerate(extra_points):
            if i % 10000 == 0:
                torch.cuda.empty_cache()

            nearest_indices, distances = find_nearest_neighbors(new_point)
            interpolated_feature = torch.zeros_like(existing_color)

            weights = distances[nearest_indices]
            mask = weights < min_dist
            weights = weights[mask]
            near_color = existing_color[nearest_indices]
            near_color = near_color[mask]

            if len(weights) == 0:
                interpolated_feature = dummy_color
                mask_points[i] = 0
            else:
                weight_sum = weights.sum()
                weights /= weight_sum
                interpolated_feature = (
                    near_color * weights[:, None, None]).sum(0)

            interpolated_colors.append(interpolated_feature.detach())
        interpolated_colors = torch.stack(interpolated_colors)

        fused_point_cloud = extra_points[mask_points]
        features = interpolated_colors[mask_points]

        fused_point_cloud = torch.concat([self.get_xyz, fused_point_cloud])
        features = torch.concat([existing_color, features])

        # dist2 = torch.clamp_min(distCUDA2(fused_point_cloud), 1e-7)
        dist2 = (knn(fused_point_cloud, 4)[:, 1:] ** 2).mean(dim=-1) # [N,]
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rotations = torch.zeros(
            (fused_point_cloud.shape[0], 4), device=self.device)
        rotations[:, 0] = 1
        opacities = inverse_sigmoid(0.1 * torch.ones(
            (fused_point_cloud.shape[0], 1), dtype=torch.float, device=self.device))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, :1].transpose(
            1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(
            features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._quaternion = nn.Parameter(rotations.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros(
            (self.get_xyz.shape[0]), device=self.device)
        self.xyz_gradient_accum = torch.zeros(
            (self.get_xyz.shape[0], 1), device=self.device)
        self.denom = torch.zeros(
            (self.get_xyz.shape[0], 1), device=self.device)

    def init_from_colmap_pcd(
        self,
        pcd: BasicPointCloud,
        sky_pcd: BasicPointCloud = None,
        init_opacity: float = 0.1,
        init_scale: float = 1.0,
    ):
        """
        Initialize from the point clouds generated by COLMAP.
        """
        points, rgbs = pcd.points, pcd.colors
        if sky_pcd is not None:
            points = np.concatenate([points, sky_pcd.points])
            rgbs = np.concatenate([rgbs, sky_pcd.colors])

        point_cloud = torch.tensor(
            np.asarray(points)).float().to(self.device)
        rgbs = torch.tensor(np.asarray(rgbs)).float().to(self.device)
        fused_color = RGB2SH(rgbs)
        features = torch.zeros(
            (fused_color.shape[0], (self.max_sh_degree + 1) ** 2, 3)
        ).float().to(self.device)
        features[:, 0, :] = fused_color

        # dist2 = torch.clamp_min(distCUDA2(
        #     torch.from_numpy(np.asarray(points)).float().to(self.device)
        # ), 0.0000001)
        dist2 = (knn(point_cloud, 4)[:, 1:] ** 2).mean(dim=-1) # [N,]
        dist2 = torch.clamp_min(dist2, 1e-7)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        quats = torch.zeros((point_cloud.shape[0], 4), device=self.device)
        quats[:, 0] = 1.0

        opacities = inverse_sigmoid(
            init_opacity * torch.ones((pcd.points.shape[0],),
                             dtype=torch.float, device=self.device)
        )
        if sky_pcd is not None:
            sky_opacities = inverse_sigmoid(
                1.0 * torch.ones((sky_pcd.points.shape[0],),
                                 dtype=torch.float, device=self.device)
            )
            opacities = torch.concat([opacities, sky_opacities])

        self._splats = {
            "means": nn.Parameter(point_cloud.requires_grad_(True)),
            "scales": nn.Parameter(scales.requires_grad_(True)),
            "quats": nn.Parameter(quats.requires_grad_(True)),
            "opacities": nn.Parameter(opacities.requires_grad_(True)),
        }

        if self.app_feat_dim is None:
            self._splats["sh0"] = nn.Parameter(
                features[:, :1, :].requires_grad_(True))
            self._splats["shN"] = nn.Parameter(
                features[:, 1:, :].requires_grad_(True))
        else:
            # features will be used for appearance and view-dependent shading.
            features = torch.rand(
                points.shape[0], self.app_feat_dim).to(self.device)
            self._splats["features"] = nn.Parameter(features.requires_grad_(True))
            colors = torch.logit(rgbs)  # [N,3]
            self._splats["colors"] = nn.Parameter(colors.requires_grad_(True))

    def init_from_external_properties(
        self,
        xyz: torch.Tensor,
        features_dc: torch.Tensor,
        features_rest: torch.Tensor,
        scaling: torch.Tensor,
        quaternion: torch.Tensor,
        opacity: torch.Tensor,
        optimizable: bool = False,
    ):
        if optimizable:
            self.set_opt_xyz(xyz)
            self.set_opt_features_dc(features_dc)
            self.set_opt_features_rest(features_rest)
            self.set_opt_raw_scaling(scaling)
            self.set_opt_raw_quaternion(quaternion)
            self.set_opt_raw_opacity(opacity)
        else:
            self.set_xyz(xyz)
            self.set_features_dc(features_dc)
            self.set_features_rest(features_rest)
            self.set_raw_scaling(scaling)
            self.set_raw_quaternion(quaternion)
            self.set_raw_opacity(opacity)
            # self.eval()

    @torch.no_grad()
    def save_ply(self, path: str):
        xyz = detach_tensor_to_numpy(self._splats["means"])
        normals = np.zeros_like(xyz)
        scale = detach_tensor_to_numpy(self._splats["scales"])
        rotation = detach_tensor_to_numpy(self._splats["quats"])
        opacity = detach_tensor_to_numpy(self._splats["opacities"]).reshape(-1, 1)
        if self.app_feat_dim is None:
            features_dc = detach_tensor_to_numpy(
                self._splats["sh0"].transpose(1, 2).flatten(1).contiguous())
            features_rest = detach_tensor_to_numpy(
                self._splats["shN"].transpose(1, 2).flatten(1).contiguous())
        else:
            features = detach_tensor_to_numpy(self._splats["features"])
            colors = detach_tensor_to_numpy(self._splats["colors"])

        dtype_full = [
            (attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        if self.app_feat_dim is None:
            attributes = np.concatenate(
                (xyz, normals, features_dc, features_rest, opacity, scale, rotation,), axis=1)
        else:
            attributes = np.concatenate(
                (xyz, normals, colors, features, opacity, scale, rotation,), axis=1)
        elements[:] = list(map(tuple, attributes))
        ply = PlyElement.describe(elements, 'vertex')
        PlyData([ply]).write(path)

    @torch.no_grad()
    def load_ply(self, path: str, optimizable: bool = False):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]['x']),
                        np.asarray(plydata.elements[0]['y']),
                        np.asarray(plydata.elements[0]['z'])), axis=1)
        opacity = np.asarray(plydata.elements[0]['opacity']).reshape(-1,)

        scale_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")
        ]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("rot")
        ]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        if self.app_feat_dim is None:
            feat_dc_names = [
                p.name for p in plydata.elements[0].properties if p.name.startswith("f_dc_")
            ]
            feat_dc_names = sorted(feat_dc_names, key = lambda x: int(x.split('_')[-1]))
            features_dc = np.zeros((xyz.shape[0], 3, 1))
            features_dc[:, 0, 0] = np.asarray(plydata.elements[0][feat_dc_names[0]])
            features_dc[:, 1, 0] = np.asarray(plydata.elements[0][feat_dc_names[1]])
            features_dc[:, 2, 0] = np.asarray(plydata.elements[0][feat_dc_names[2]])
            features_dc = torch.tensor(features_dc).transpose(1, 2).contiguous()

            feat_rest_names = [
                p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")
            ]
            assert len(feat_rest_names) == 3*(self.max_sh_degree + 1) ** 2 - 3
            feat_rest_names = sorted(feat_rest_names, key = lambda x: int(x.split('_')[-1]))
            features_rest = np.zeros((xyz.shape[0], len(feat_rest_names)))
            for idx, attr_name in enumerate(feat_rest_names):
                features_rest[:, idx] = np.asarray(plydata.elements[0][attr_name])
            features_rest = torch.tensor(features_rest)
            features_rest = features_rest.reshape(
                (xyz.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1)).transpose(1, 2).contiguous()
        else:
            color_names = [
                p.name for p in plydata.elements[0].properties if p.name.startswith("color_")
            ]
            color_names = sorted(color_names, key = lambda x: int(x.split('_')[-1]))
            colors = np.zeros((xyz.shape[0], len(color_names)))
            for idx, attr_name in enumerate(color_names):
                colors[:, idx] = np.asarray(plydata.elements[0][attr_name])
            feature_names = [
                p.name for p in plydata.elements[0].properties if p.name.startswith("feature_")
            ]
            feature_names = sorted(feature_names, key = lambda x: int(x.split('_')[-1]))
            features = np.zeros((xyz.shape[0], len(feature_names)))
            for idx, attr_name in enumerate(feature_names):
                features[:, idx] = np.asarray(plydata.elements[0][attr_name])
            features_dc = features

        if optimizable:
            self.set_opt_xyz(torch.tensor(xyz).float().to(self.device))
            self.set_opt_raw_opacity(torch.tensor(opacity).float().to(self.device))
            self.set_opt_raw_scaling(torch.tensor(scales).float().to(self.device))
            self.set_opt_raw_quaternion(torch.tensor(rots).float().to(self.device))
            if self.app_feat_dim is None:
                self.set_opt_features_dc(features_dc.float().to(self.device))
                self.set_opt_features_rest(features_rest.float().to(self.device))
            else:
                self.set_opt_features_dc(torch.tensor(features_dc).float().to(self.device))
                self.set_opt_colors(torch.tensor(colors).float().to(self.device))
        else:
            self.set_xyz(torch.tensor(xyz).float().to(self.device))
            self.set_raw_opacity(torch.tensor(opacity).float().to(self.device))
            self.set_raw_scaling(torch.tensor(scales).float().to(self.device))
            self.set_raw_quaternion(torch.tensor(rots).float().to(self.device))
            if self.app_feat_dim is None:
                self.set_features_dc(features_dc.float().to(self.device))
                self.set_features_rest(features_rest.float().to(self.device))
            else:
                self.set_features_dc(torch.tensor(features_dc).float().to(self.device))
                self.set_colors(torch.tensor(colors).float().to(self.device))

    @torch.no_grad()
    def save_colmap_ply(self, path: str):
        xyz = self.get_xyz
        if self.app_feat_dim is None:
            shs_view = (
                self.get_features.transpose(1, 2)
                .view(-1, 3, (self.max_sh_degree + 1) ** 2)
            )
            sh2rgb = eval_sh(
                deg=0,
                sh=shs_view,
                dirs=None,
            )
            rgbs = torch.clamp_min(sh2rgb + 0.5, 0.0).cpu().detach().numpy() * 255
        else:
            rgbs = self.get_colors.cpu().detach() * 255
            rgbs = rgbs.to(torch.uint8).numpy()

        num_points = xyz.shape[0]
        file = open(path, 'w', encoding="utf-8")
        file.write("# 3D point list with one line of data per point:\n")
        file.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, " +
                   "TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        file.write(f"# Number of points: {num_points}, mean track length: 0\n")
        for i in range(num_points):
            file.write(f'{i} ')
            file.write(f'{xyz[i][0]} {xyz[i][1]} {xyz[i][2]} ' +
                       f'{rgbs[i][0]} {rgbs[i][1]} {rgbs[i][2]} 0 \n')

        file.close()

    @torch.no_grad()
    def save_splat(self, output_path: str = "", rgb: torch.Tensor = None):
        buffer = BytesIO()

        xyz = self.get_xyz.detach().cpu().numpy()
        scale = self.get_raw_scaling.detach().cpu().numpy()
        opacity = self.get_raw_opacity.detach().cpu().numpy()
        quaternion = self.get_raw_quaternion.detach().cpu().numpy()
        if self.app_feat_dim is None:
            features_dc = self.get_features_dc.detach().cpu().squeeze(dim=1).numpy()
        else:
            rgb = torch.sigmoid(rgb + self.get_colors).squeeze(0)
            colors = rgb.detach().cpu().numpy()

        sorted_indices = np.argsort(
            -np.exp(scale[:, 0] + scale[:, 1] + scale[:, 2])
            / (1 + np.exp(opacity[:]))
        )
        SH_C0 = 0.28209479177387814

        pbar = tqdm.trange(len(sorted_indices), desc="Saving Splat file")
        for idx in sorted_indices:
            position = np.array(
                [xyz[idx][0], xyz[idx][1], xyz[idx][2]], dtype=np.float32)
            scales = np.exp(
                np.array([scale[idx][0], scale[idx][1],
                         scale[idx][2]], dtype=np.float32)
            )
            rot = np.array(
                [quaternion[idx][0], quaternion[idx][1],
                    quaternion[idx][2], quaternion[idx][3]],
                dtype=np.float32
            )
            if self.app_feat_dim is None:
                color = np.array([
                    0.5 + SH_C0 * features_dc[idx][0],
                    0.5 + SH_C0 * features_dc[idx][1],
                    0.5 + SH_C0 * features_dc[idx][2],
                    1 / (1 + np.exp(-opacity[idx])),
                ])
            else:
                color = np.array([
                    colors[idx][0], colors[idx][1], colors[idx][2],
                    1 / (1 + np.exp(-opacity[idx])),
                ])
            buffer.write(position.tobytes())
            buffer.write(scales.tobytes())
            buffer.write((color * 255).clip(0, 255).astype(np.uint8).tobytes())
            buffer.write(
                ((rot / np.linalg.norm(rot)) * 128 + 128)
                .clip(0, 255).astype(np.uint8).tobytes()
            )
            pbar.update(1)

        with open(output_path, "wb") as file:
            file.write(buffer.getvalue())

    def to(self, device: str = "cuda"):
        new_gaussians = GaussianSplatModel(
            self.max_sh_degree, self.percent_dense, self.app_feat_dim
        )
        new_gaussians.set_xyz(self.get_xyz.to(device))
        new_gaussians.set_raw_scaling(self.get_raw_scaling.to(device))
        new_gaussians.set_raw_quaternion(self.get_raw_quaternion.to(device))
        new_gaussians.set_raw_opacity(self.get_raw_opacity.to(device))
        new_gaussians.set_features_dc(self.get_features_dc.to(device))
        if self.app_feat_dim is None:
            new_gaussians.set_features_rest(self.get_features_rest.to(device))
        else:
            new_gaussians.set_colors(self.get_colors.to(device))
        new_gaussians.active_sh_degree = self.active_sh_degree

        return new_gaussians
