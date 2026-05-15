# pylint: disable=[E1101,W0621]

import copy
import os
import time
import json
from typing import List, Literal
import tqdm

import torch
import torch.nn.functional as F

from omegaconf import OmegaConf

from conerf.base.checkpoint_manager import CheckPointManager
from conerf.evaluators.evaluator import (
    Evaluator, compute_psnr, compute_lpips, compute_ssim, color_correct
)
from conerf.evaluators.gaussian_splatting_evaluator import GaussianSplatEvaluator
from conerf.geometry.camera import Camera
from conerf.gaussian_fields.app_embed import AppearanceOptModule
from conerf.gaussian_fields.gaussian_splat_model import GaussianSplatModel
from conerf.utils.utils import save_images, get_subdirs, colorize
from PIL import Image
from fused_ssim import fused_ssim
from conerf.datasets.utils import create_dataset
from conerf.utils.trajs_utils import ViewSelection, PoseFilter
from difix3d.pipeline_difix import DifixPipeline
from difix3d.utils import CameraPoseInterpolator
from torchvision import transforms


def calculate_wiou(val_masks: list[torch.Tensor], 
                   train_masks: list[torch.Tensor], 
                   batch_size: int = 8) -> torch.Tensor:
    """
    Calculates a pairwise Weighted Intersection over Union (WIoU) matrix 
    between two lists of weighted masks: W_val_masks (N) and W_train_masks (M).
    
    The resulting matrix is N x M, where [i, j] is WIoU(W_val_masks[i], W_train_masks[j]).

    Args:
        W_val_masks (list[torch.Tensor]): The 'query' masks (N masks).
        W_train_masks (list[torch.Tensor]): The 'reference' masks (M masks).
        batch_size (int): Batch size for processing query mask groups.

    Returns:
        torch.Tensor: An [N, M] WIoU matrix.
    """
    N = len(val_masks)
    M = len(train_masks)
    
    if N == 0 or M == 0:
        return torch.empty((N, M), dtype=torch.float32)

    all_val = torch.stack(val_masks, dim=0).float().flatten(start_dim=1)  # Shape: [N, L]
    all_train = torch.stack(train_masks, dim=0).float().flatten(start_dim=1) # Shape: [M, L]

    device = all_val.device
    wiou_matrix = torch.zeros((N, M), device=device, dtype=torch.float32)
    
    sum_W_val = all_val.sum(dim=1)  # Shape: [N]
    sum_W_train = all_train.sum(dim=1) # Shape: [M]

    for i in range(0, N, batch_size):
        i_end = min(i + batch_size, N)
        W_A = all_val[i:i_end]   # Query masks (Batch B, Shape: [B, L])
        W_B = all_train          # Reference masks (M, Shape: [M, L])

        # [B, 1, L] vs [1, M, L] -> min([B, M, L]) -> sum(dim=2) -> [B, M]
        weighted_intersection_batch = torch.min(W_A[:, None, :], W_B[None, :, :]).sum(dim=2) 
        sum_A_batch = sum_W_val[i:i_end]  # Shape: [B]
        sum_B = sum_W_train               # Shape: [M]
        
        # [B, 1] + [1, M] - [B, M] -> [B, M]
        weighted_union_batch = sum_A_batch.unsqueeze(1) + sum_B.unsqueeze(0) - weighted_intersection_batch 
        wiou_batch = weighted_intersection_batch / (weighted_union_batch + 1e-6)  # Shape: [B, M]
        wiou_matrix[i:i_end, :] = wiou_batch

    return wiou_matrix

def get_val_to_train_wiou(wiou_matrix: torch.Tensor, 
                          train_node_ids: list, 
                          keep_max_only: bool = True) -> dict:
    N, M = wiou_matrix.shape
    max_wiou_values, max_wiou_indices = torch.max(wiou_matrix, dim=1) # Both shape [N]
    
    result_map = {}
    
    for i in range(N):
        if keep_max_only:
            max_index = max_wiou_indices[i].item()
            best_train_id = train_node_ids[max_index]
            max_value = max_wiou_values[i].item()
            
            result_map[str(i)] = (best_train_id, max_value)
            
        else:
            sorted_wiou, sorted_indices = torch.sort(wiou_matrix[i, :], descending=True)
            
            all_pairs = []
            for j in range(M):
                train_idx = sorted_indices[j].item()
                train_id = train_node_ids[train_idx]
                wiou_value = sorted_wiou[j].item()
                
                if wiou_value > 1e-4:
                    all_pairs.append((train_id, wiou_value))
                else:
                    break
                    
            result_map[str(i)] = all_pairs
            
    return result_map

class ActiveEvaluator(GaussianSplatEvaluator):
    """Class for evaluating NeRF models."""

    def __init__(
        self,
        config: OmegaConf,
        load_train_data: bool = True,
        trainset=None,
        load_val_data: bool = True,
        valset=None,
        load_test_data: bool = False,
        testset=None,
        models: List = None,
        meta_data: List = None,
        verbose: bool = False,
        device: str = "cuda",
    ) -> None:
        super().__init__(
            config,
            load_train_data,
            trainset,
            load_val_data,
            valset,
            load_test_data,
            testset,
            models,
            meta_data,
            verbose,
            device,
        )

    def load_dataset(
        self,
        load_train_data: bool = False,
        load_val_data: bool = True,
        load_test_data: bool = False,
    ):
        """Loading train dataset or validation dataset if required."""
        if load_train_data:
            self.train_dataset = create_dataset(
                config=self.config,
                split=self.config.dataset.train_split,
                num_rays=None,
                apply_mask=self.config.dataset.apply_mask,
                device=self.device
            )

        if load_val_data:
            self.val_dataset = create_dataset(
                config=self.config,
                split=self.config.dataset.val_split,
                num_rays=None,
                apply_mask=self.config.dataset.apply_mask,
                device=self.device,
            )

        if load_test_data:
            self.test_dataset = create_dataset(
                config=self.config,
                split="test",
                num_rays=None,
                apply_mask=self.config.dataset.apply_mask,
                device=self.device,
            )

    def setup_denoiser(self):
        ## cameras clustering & visualization
        view_selector = ViewSelection(self.models[0])
        self.pose_filter = PoseFilter(view_selector, camera_params=self.train_dataset.cameras[0], 
                                      is_uncertainty=False, 
                                      ) 
        ## initial DIFIX
        self.interpolator = CameraPoseInterpolator(rotation_weight=1.0, translation_weight=1.0)
        self.difix = DifixPipeline.from_pretrained("nvidia/difix_ref", trust_remote_code=True)
        self.difix.set_progress_bar_config(disable=True)
        self.difix.to("cuda")

    def eval_denoise(
            self,
            iteration: int = None,
            color_correct: bool = False,
            split: Literal["val", "test"] = "val",):
        self.setup_denoiser()
        
        metrics = dict()
        if split == "val":
            assert self.val_dataset is not None
            dataset = self.val_dataset
        elif split == "test":
            assert self.test_dataset is not None
            dataset = self.test_dataset
        else:
            if self.verbose:
                print(f'[WARNING] {split} set does not exist!')
            return
        cameras = dataset.cameras

        # calculate wiou mask
        train_c2ws = torch.stack([cam.cam_to_world for cam in self.train_dataset.cameras])
        val_c2ws = torch.stack([cam.cam_to_world for cam in cameras])
        poses = torch.cat([train_c2ws, val_c2ws], dim=0) # N, 4, 4
        N_train = len(train_c2ws)
        # poses scoring and matching
        all_masks = []
        for pose in tqdm.tqdm(poses, desc="Evaluate pose quality"):
            _, mask = self.pose_filter._score_pose_and_calculate_mask(pose)
            all_masks.append(mask)
        iou_matrix = calculate_wiou(all_masks[N_train:], all_masks[:N_train])
        matched_trainid = get_val_to_train_wiou(iou_matrix, list(range(N_train)))

        eval_dir = os.path.join(self.eval_dir, split)
        os.makedirs(eval_dir, exist_ok=True)


        meta_data = self.meta_data[0]
        meta_data["split"] = split

        image_dir = os.path.join(self.eval_dir, split, "images")
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(os.path.join(self.eval_dir, split, "images", "rgb_denoise"), exist_ok=True)

        pbar = tqdm.trange(
            len(cameras), desc=f"Validating {self.config.expname}", leave=False
        )
        psnrs, ssims, lpips = {}, {}, {}
        for i in range(len(cameras)):  # pylint: disable=C0200
            train_id, max_iou = matched_trainid[str(i)]
            psnrs[i], ssims[i], lpips[i] = self.run_difix(
                cameras[i], meta_data, image_dir, i, train_id, color_correct=color_correct
            )

            pbar.update(1)
        
        avg_psnr = sum(psnrs.values()) / len(psnrs)
        avg_ssim = sum(ssims.values()) / len(ssims)
        avg_lpips = sum(lpips.values()) / len(lpips)
        metrics["global"] = {
            'iteration': iteration,
            'all_psnr': psnrs,
            'all_ssim': ssims,
            'all_lpips': lpips,
            'psnr': avg_psnr,
            'ssim': avg_ssim,
            'lpips': avg_lpips,
        }
        metric_file = os.path.join(eval_dir, 'metrics_denoise.json')
        json_obj = json.dumps(metrics, indent=4)
        if self.verbose:
            print(f'Saving metrics to {metric_file}')
        with open(metric_file, 'a', encoding='utf-8') as json_file:
            json_file.write(json_obj)

        
    @torch.no_grad()
    def run_difix(self, camera, meta_data, image_dir, i, ref_i, color_correct=False):
        image_path = os.path.join(image_dir, "rgb_test", f"{i:03d}.png")
        ref_path = self.train_dataset.cameras[ref_i].image_path

        assert os.path.exists(image_path), f"{image_path} No Found"
        image = Image.open(image_path).convert("RGB")

        assert os.path.exists(ref_path), f"{ref_path} No Found"
        ref_image = Image.open(ref_path).convert("RGB")
        ref_image = ref_image.resize(image.size, Image.BILINEAR)
        output_image = self.difix(prompt="remove degradation", image=image, ref_image=ref_image, 
                num_images_per_prompt=1, num_inference_steps=1, timesteps=[199], guidance_scale=0.0).images[0]
        output_image = output_image.resize(image.size, Image.LANCZOS)
        output_image.save(f"{image_dir}/rgb_denoise/{i:03d}.png")

        # calculate metrics
        colors = transforms.ToTensor()(output_image)
        pixels = camera.image  # [height, width, RGB]

        if color_correct:
            colors_cc = color_correct(colors.permute(
                1, 2, 0).numpy(), pixels.numpy())
            colors_cc = torch.from_numpy(colors_cc).permute(2, 0, 1)
            colors_cc = colors
        else:
            colors_cc = colors
        
        if meta_data["split"] == "val":
            pixels = pixels[None, ...].to(self.device).permute(0, 3, 1, 2)
            colors_cc = colors_cc[None, ...].to(self.device)
            psnr = compute_psnr(pixels, colors_cc).item()
            ssim = compute_ssim(pixels, colors_cc)
            lpips = compute_lpips(self.lpips_loss, pixels, colors_cc)
        else:
            psnr, ssim, lpips = 0, 0, 0

        return psnr, ssim, lpips
