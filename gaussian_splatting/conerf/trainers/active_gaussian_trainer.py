# pylint: disable=[E1101,E1102,R1719,W0201]

import os
import random
import copy
from typing import List, Tuple
from omegaconf import OmegaConf

import torch
import torch.nn.functional as F
import numpy as np
import json

from conerf.base.model_base import ModelBase
from conerf.base.task_queue import ImageReader
from conerf.render.gaussian_render import render_gsplat
from conerf.trainers.gaussian_trainer import GaussianSplatTrainer

from fused_ssim import fused_ssim


class RandSelector(torch.nn.Module):
    def __init__(self, num_views: int = 1, seed: int = 44):
        super().__init__()

        self.num_views = num_views
        self.seed = seed

    def next_best_views(self, candidate_cameras) -> List[int]:
        index = list(range(len(candidate_cameras)))
        random.Random(self.seed).shuffle(index)

        return index[:self.num_views]


class ScoreSelector(torch.nn.Module):
    def __init__(
        self,
        num_views: int = 1,
        seed: int = 44,
        score_list: str = "conf.txt",
    ):
        super().__init__()

        self.num_views = num_views
        self.seed = seed

        raw_scores = {}
        with open(score_list, "r") as f:
            for line in f:
                parts = line.strip().split()
                image_id = parts[0]
                uncertainty = float(parts[1])
                depth_score = float(parts[2])
                image_score = float(parts[3])

                normalized_uncertainty = (
                    uncertainty - np.min(uncertainty)
                ) / (np.max(uncertainty) - np.min(uncertainty) + 1e-6)
                raw_scores[image_id] = normalized_uncertainty + \
                    depth_score + (1 - image_score) * 3
        self.scores = raw_scores

    def next_best_views(self, candidate_cameras) -> List[int]:
        if len(candidate_cameras) == 0:
            return []
        camera_indices = [cam.image_index for cam in candidate_cameras]
        current_scores = [self.scores[cid] for cid in camera_indices]
        index = list(range(len(candidate_cameras)))

        sorted_index = sorted(
            index,
            key=lambda i: current_scores[i],
            reverse=True
        )
        return sorted_index[:self.num_views]


def get_loss_decay_factor(value) -> torch.Tensor:
    if isinstance(value, float):
        b = torch.tensor(value).float()
    else:
        b = torch.from_numpy(value).float()
    decay_factors = torch.empty_like(b)

    mask_low = b < 0.3
    decay_factors[mask_low] = 0.6
    mask_high = b > 0.5
    decay_factors[mask_high] = 0.1

    mask_transition = ~(mask_low | mask_high)
    b_transition = b[mask_transition]

    start_val = 0.6
    end_val = 0.1
    start_point = 0.3
    end_point = 0.5
    normalized_b = (b_transition - start_point) / (end_point - start_point)
    interpolated_factors = start_val + (end_val - start_val) * normalized_b
    decay_factors[mask_transition] = interpolated_factors

    return decay_factors


class CertaintySelector(torch.nn.Module):
    def __init__(
        self,
        num_views: int = 1,
        seed: int = 44,
        score_list: str = "conf.txt",
        vg: str = "view_graph.json"
    ):
        super().__init__()

        self.num_views = num_views
        self.seed = seed

        if os.path.exists(vg):
            view_graph = json.load(open(vg, "r"))
            wiou = {
                nk: sum(item[1] for item in nv if 'fv' not in str(item[0]))
                for nk, nv in view_graph.items() if "fv" in nk
            }

        raw_scores = {}
        loss_decay = {}
        with open(score_list, "r") as f:
            for line in f:
                parts = line.strip().split()
                image_id = parts[0]
                image_score = float(parts[3])
                loss_decay[image_id] = get_loss_decay_factor(image_score)
                raw_scores[image_id] = (1 - image_score) + (1 - wiou[image_id])
        self.scores = raw_scores
        self.decay = loss_decay

    def next_best_views(self, candidate_cameras) -> Tuple[List[int], List[int]]:
        if len(candidate_cameras) == 0:
            return [], []

        camera_indices = [cam.image_index for cam in candidate_cameras]
        current_scores = [self.scores[cid] for cid in camera_indices]
        current_decay = [self.decay[cid] for cid in camera_indices]
        index = list(range(len(candidate_cameras)))

        sorted_index = sorted(
            index,
            key=lambda i: current_scores[i],
            reverse=True
        )
        selected = sorted_index[:self.num_views]

        return selected, current_decay


class ActiveGaussianSplatTrainer(GaussianSplatTrainer):
    """
    Trainer for Active 3D Gaussian Splatting model.
    """

    def __init__(
        self,
        config: OmegaConf,
        prefetch_dataset: bool = True,
        trainset=None,
        valset=None,
        model: ModelBase = None,
        appear_embedding: torch.nn.Module = None,
        block_id: int = None,
        device_id: int = 0,
    ) -> None:

        self.output_path = os.path.join(
            config.dataset.output_dir, config.expname)
        # Load free views.
        sampler_dir = getattr(config.dataset, "sampler_dir", None)
        if sampler_dir:
            sampled_dir = os.path.join(sampler_dir, config.expname)
        else:
            sampled_dir = self.output_path
        free_view_path = f"{sampled_dir}/renders/cameras_difix.pt"
        assert os.path.exists(free_view_path), \
            f"Free views file does not exist: {free_view_path}!"
        freeviews = torch.load(free_view_path, weights_only=False)

        self.freeviews = []
        for view in freeviews:
            if os.path.exists(view.image_path):
                if isinstance(view.image_index, int):
                    view.image_index = "fv_" + str(view.image_index)
                self.freeviews.append(view)
        print(f"Collect {len(self.freeviews)} free views.")

        super().__init__(config, prefetch_dataset,
                         trainset, valset, model, appear_embedding, block_id, device_id)

        self.active_method = CertaintySelector(
            num_views=config.trainer.num_add_peraug,
            score_list=f"{sampled_dir}/renders/conf.txt",
            vg=f"{sampled_dir}/renders/view_graph.json"
        )

    def setup_training_params(self):
        super().setup_training_params()

        # Override train cameras.
        self.train_cameras = self.train_dataset.cameras.copy()

        self.train_camera_idxs = [
            camera.image_index for camera in self.train_cameras + self.freeviews]
        self.image_idx_to_global_index = {
            idx: i for i, idx in enumerate(self.train_camera_idxs)
        }

        self.aug_iters = self.config.trainer.aug_iterations
        self.loss_decay = {}

    def train_iteration(self, data_batch) -> None:  # pylint: disable=W0613
        self.gaussians.train()
        self.update_learning_rate()

        # Increase the levels of SH up to a maximum degree.
        if self.iteration % 1000 == 0:
            self.gaussians.increase_SH_degree()

        # Training finished and safely exit.
        if (self.iteration - 1) >= self.config.trainer.max_iterations:
            self.image_reader.safe_shutdown()
            return torch.zeros(1), torch.zeros(1)

        # Pick a random camera.
        if self.image_reader is None or (not self.image_reader.has_next()):
            if len(self.aug_iters) > 0 and self.iteration > self.aug_iters[0]:
                self.add_freeviews()
                self.aug_iters = self.aug_iters[1:]

            # add_cam = copy.deepcopy(self.train_dataset.cameras[0])
            # self.train_cameras.append(add_cam)

            random.shuffle(self.train_cameras)
            image_list = [camera.image_path for camera in self.train_cameras]
            # Add depth list if exists!
            depth_list = [camera.depth_path for camera in self.train_cameras] \
                if self.config.dataset.get("load_depth", False) else None
            # Add sky mask list if exists!
            sky_mask_list = [camera.mask_path for camera in self.train_cameras] \
                if self.config.dataset.get("load_sky_mask", False) else None

            if self.image_reader is None:
                self.image_reader = ImageReader(
                    num_channels=self.config.dataset.get('num_channels', 3),
                    max_num_threads=self.config.trainer.get('num_workers', 8),
                    max_size=len(image_list),
                    image_list=image_list,
                    depth_list=depth_list,
                    mask_list=sky_mask_list,
                )
            else:
                self.image_reader.image_list = image_list
                self.image_reader.depth_list = depth_list
                self.image_reader.mask_list = sky_mask_list

            self.image_reader.start_loading()

        image_index, image, inv_depth, _, __ = self.image_reader.get_next()
        camera = copy.deepcopy(self.train_cameras[image_index])
        global_index = self.image_idx_to_global_index[camera.image_index]

        camera.image = image
        if inv_depth is not None:
            camera.inv_depth = inv_depth
            if hasattr(self, 'image_index_to_depth_params'):
                camera.depth_params = self.image_index_to_depth_params.get(
                    camera.image_index, {}
                )
            camera.check_depth()
        resolution = self.training_resolution()
        camera_origin = camera.copy_to_device(self.device) \
            if self.mask is not None else None
        camera = camera.downsample(resolution).copy_to_device(self.device)

        self.scalars_to_log['train/resolution'] = resolution

        # Since we only update on the copy of cameras, the update won't
        # be accumulated continuously on the same cameras.
        if self.optimize_camera_poses and (camera.image_index != 0) and \
           (self.iteration > self.config.geometry.opt_pose_start_iter):
            image_index = camera.image_index

        precompute_colors = None
        if self.app_module is not None:
            dirs = self.gaussians.get_xyz - camera.camera_center.repeat(
                self.gaussians.get_features.shape[0], 1)
            precompute_colors = self.app_module(
                features=self.gaussians.get_features,
                embed_ids=torch.tensor([global_index], device=self.device),
                dirs=dirs[None],
                sh_degree=self.gaussians.max_sh_degree,
            )
            precompute_colors = precompute_colors + self.gaussians.get_colors
            precompute_colors = torch.sigmoid(precompute_colors)

        render_results = render_gsplat(
            gaussian_splat_model=self.gaussians,
            viewpoint_camera=camera,
            pipeline_config=self.config.pipeline,
            bkgd_color=self.color_bkgd,
            anti_aliasing=self.config.texture.anti_aliasing,
            separate_sh=False,  # True,
            # use_trained_exposure=self.config.appearance.use_trained_exposure,
            exposure=self.exposure[global_index] if self.exposure is not None else None,
            override_color=precompute_colors,
            device=self.device,
        )
        colors, screen_space_points, visibility_filter, radii = (
            render_results["rendered_image"],
            render_results["screen_space_points"],
            render_results["visibility_filter"],
            render_results["radii"],
        )

        self.densify_strategy.step_pre_backward(
            params=self.gaussians.get_param_dict(),
            optimizers=self.get_gs_primitive_optimizers(),
            state=self.strategy_state,
            step=self.iteration,
            info=render_results["info"],
        )

        # Compute loss.
        lambda_dssim = self.config.loss.lambda_dssim
        lambda_mask = self.config.loss.lambda_mask
        pixels = camera.image.permute(2, 0, 1)  # [RGB, height, width]
        # loss_ssim = ssim(pixels, colors)
        loss_ssim = fused_ssim(colors.unsqueeze(0), pixels.unsqueeze(0))
        if self.mask is not None:
            image_size = camera.image.shape[:-1]
            camera = camera_origin.downsample(32).copy_to_device(self.device)
            mask = self.mask(camera.image.permute(2, 0, 1),
                             camera.image_index, image_size)
            loss_rgb_l1 = F.l1_loss(colors * mask, pixels)
            loss = (1.0 - lambda_dssim) * loss_rgb_l1 + \
                lambda_dssim * (1.0 - loss_ssim) + \
                lambda_mask * torch.mean((mask - 1) **
                                         2.)  # Regularization for mask
        else:
            loss_rgb_l1 = F.l1_loss(colors, pixels)
            loss = (1.0 - lambda_dssim) * loss_rgb_l1 + \
                lambda_dssim * (1.0 - loss_ssim)

        # Scaling loss.
        lambda_scale = self.config.loss.lambda_scale if self.use_mcmc else 0
        loss_scaling = torch.abs(render_results["scaling"]).mean()
        loss += lambda_scale * loss_scaling

        # Opacity Loss.
        lambda_opacity = self.config.loss.lambda_opacity if self.use_mcmc else 0
        loss_opacity = torch.abs(self.gaussians.get_opacity).mean()
        loss += lambda_opacity * loss_opacity

        # Depth loss.
        if self.depth_scheduler is not None and (
            self.depth_scheduler(self.iteration) > 0 and camera.depth_reliable
        ):
            inv_depth = 1.0 / (render_results["depth"] + 1e-6)
            mono_inv_depth = camera.inv_depth
            l1_depth_pure = torch.abs(inv_depth - mono_inv_depth).mean()
            l1_loss_depth = self.depth_scheduler(
                self.iteration) * l1_depth_pure
            loss += l1_loss_depth
            self.scalars_to_log["train/depth_loss"] = l1_loss_depth.detach().item()

        if self.admm_enabled:
            loss = self.add_admm_penalties(loss)

        if isinstance(camera.image_index, str):
            if camera.image_index in self.loss_decay.keys():
                decay_w = self.loss_decay[camera.image_index]
                loss = decay_w * loss
            else:
                loss = self.config.loss.lambda_novel_data * loss

        loss.backward()

        self.ema_loss = 0.4 * loss.detach().item() + 0.6 * \
            self.ema_loss  # pylint: disable=W0201

        mse = F.mse_loss(colors, pixels)
        psnr = -10.0 * torch.log(mse) / np.log(10.0)

        # training statistics.
        self.scalars_to_log["train/psnr"] = psnr.detach().item()
        self.scalars_to_log["train/loss"] = loss.detach().item()
        self.scalars_to_log["train/l1_loss"] = loss_rgb_l1.detach().item()
        self.scalars_to_log["train/scale_loss"] = loss_scaling.detach().item()
        self.scalars_to_log["train/opacity_loss"] = loss_opacity.detach().item()
        self.scalars_to_log["train/ema_loss"] = self.ema_loss
        self.scalars_to_log["train/points"] = self.gaussians.get_xyz.shape[0]

        # Optimizer step.
        visibility_mask = (radii > 0).any(0)
        self.optimizers.step_and_zero_grad(
            set_to_none=True,
            visibility_mask=visibility_mask,
        )
        self.optimizers.schedule()

        if self.config.geometry.get("densify_strategy", "mcmc") == "mcmc":
            self.densify_strategy.step_post_backward(
                params=self.gaussians.get_param_dict(),
                optimizers=self.get_gs_primitive_optimizers(),
                state=self.strategy_state,
                step=self.iteration,
                info=render_results["info"],
                lr=self.xyz_scheduler(self.iteration),
            )
        else:
            self.densify_strategy.step_post_backward(
                params=self.gaussians.get_param_dict(),
                optimizers=self.get_gs_primitive_optimizers(),
                state=self.strategy_state,
                step=self.iteration,
                info=render_results["info"],
                packed=False,
            )

        if self.iteration % self.config.trainer.n_checkpoint == 0:
            self.compose_state_dicts()

        # Update camera pose.
        if self.optimize_camera_poses and (camera.image_index != 0) and \
           (self.iteration > self.config.geometry.opt_pose_start_iter):
            self.train_dataset.cameras[image_index].update_camera_pose()

        if self.iteration in [aug_i - 1 for aug_i in self.config.trainer.aug_iterations]:
            self.add_freeviews()

        return psnr, loss_rgb_l1

    def add_freeviews(self):
        selected_index, loss_decay = self.active_method.next_best_views(
            self.freeviews)
        for i, cam_to_add in enumerate(self.freeviews):
            if i in selected_index:
                self.loss_decay[cam_to_add.image_index] = loss_decay[i]
                self.train_cameras.append(cam_to_add)
                self.freeviews.remove(cam_to_add)
        print(
            f"Select novel views {selected_index} with uncertainty, remain {len(self.freeviews)}")
