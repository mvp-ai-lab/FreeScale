from typing import Tuple

import torch
import tqdm
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class ViewSelection:
    def __init__(self, gaussians, grid_resolution=128, margin=0.1):
        self.resolution = grid_resolution
        self.means = gaussians.get_xyz.detach()
        self.alpha = torch.sigmoid(gaussians.get_opacity.detach())
        self.scales = gaussians.get_scaling.detach()
        self.quats = gaussians.get_quaternion.detach()
        self.volume = torch.prod(torch.exp(self.scales), dim=1)  # Shape (N,)
        self.device = self.alpha.device

        """Calculates the bounding box and voxel size for the scene."""
        bbox_min = torch.min(self.means, dim=0)[0]
        bbox_max = torch.max(self.means, dim=0)[0]

        # Add margin
        bbox_size = bbox_max - bbox_min
        bbox_min -= bbox_size * margin
        bbox_max += bbox_size * margin

        self.bbox_min = bbox_min
        self.bbox_max = bbox_max
        self.voxel_size = (self.bbox_max - self.bbox_min) / self.resolution

    def _build_grid(self, scores) -> torch.Tensor:
        """
        Generic function to build a grid by scattering scores.

        Args:
            scores: A tensor of shape (N,) with the score for each Gaussian.

        Returns:
            torch.Tensor: A 3D grid of shape (res, res, res) with accumulated scores.
        """
        grid = torch.zeros((self.resolution, self.resolution, self.resolution),
                           device=self.device, dtype=torch.float32)

        # Normalize means to be within [0, 1] and then scale to grid indices
        normalized_means = (self.means - self.bbox_min) / \
            (self.bbox_max - self.bbox_min)
        voxel_indices_float = normalized_means * (self.resolution - 1)
        voxel_indices = torch.clamp(
            voxel_indices_float.long(), 0, self.resolution - 1)

        # Flatten 3D indices to 1D for scatter_add_
        flat_indices = voxel_indices[:, 0] * self.resolution * self.resolution + \
            voxel_indices[:, 1] * self.resolution + \
            voxel_indices[:, 2]
        grid.view(-1).scatter_add_(0, flat_indices.to(self.device), scores)

        return grid

    def build_uncertainty_grid(self) -> torch.Tensor:
        """Builds a grid where high values indicate high uncertainty (e.g., floaters)."""
        # Score = (1 - alpha) * volume
        # uncertainty_scores = (1 - self.alpha) * self.volume
        # torch.sum(self.scales, dim=1)  # Use sum of scales for uncertainty
        uncertainty_scores = (1 - self.alpha) * self.volume
        return self._build_grid(uncertainty_scores)

    def build_certainty_grid(self) -> torch.Tensor:
        """Builds a grid where high values indicate high certainty (well-reconstructed surfaces)."""
        # Score = alpha / (volume + epsilon)
        certainty_scores = self.alpha / (self.volume + 1e-8)
        return self._build_grid(certainty_scores)

    def build_occupancy_grid(self) -> torch.Tensor:
        """Builds a grid where high values indicate high occupancy."""
        # Score = alpha
        occupancy_scores = self.alpha
        grid = self._build_grid(occupancy_scores)

        # normalization
        min_val = grid.min()
        non_zero_grid = grid[grid > 1e-8]
        if non_zero_grid.numel() > 0:
            quantile_max = torch.quantile(non_zero_grid, 0.9)
        else:
            quantile_max = min_val
        max_val_robust = quantile_max

        if max_val_robust > min_val:
            grid = torch.clamp(grid, min=min_val, max=max_val_robust)
            grid = (grid - min_val) / (max_val_robust - min_val)
        return grid

    def get_world_coord(self, indices) -> torch.Tensor:
        world_coords = self.bbox_min + indices * self.voxel_size
        return world_coords  # [N_voxels, 3]


class PoseFilter:
    def __init__(
        self,
        view_selector: ViewSelection,
        camera_params: dict,
        pcl=None,
    ):
        """
        Initializes the PoseFilter.

        Args:
            view_selector (ViewSelection): An initialized ViewSelection object.
            camera_params (dict): Camera class.
            device (str): The device to run computations on.
        """
        self.selector = view_selector
        self.device = view_selector.device
        self.points3d = pcl

        # Build and store the grids
        self.uncertainty_grid = self.selector.build_uncertainty_grid().unsqueeze(
            0).unsqueeze(0)  # Shape [1, 1, D, H, W] for grid_sample
        self.certainty_grid = self.selector.build_certainty_grid().unsqueeze(0).unsqueeze(0)
        self.occupancy_grid = self.selector.build_occupancy_grid().unsqueeze(0).unsqueeze(0)
        # get non-zero certainty coordinations
        grid_indices = torch.nonzero(
            self.certainty_grid.squeeze(), as_tuple=False)
        self.certainty_values = self.certainty_grid.squeeze()[grid_indices[:, 0],
                                                              grid_indices[:, 1],
                                                              grid_indices[:, 2]].unsqueeze(1)  # [N_voxels, 1]
        self.certainty_world_coords = self.selector.get_world_coord(
            grid_indices)

        # Camera parameters
        self.H = camera_params.height
        self.W = camera_params.width
        self.focal_x = camera_params.fx
        self.focal_y = camera_params.fy
        self.K = camera_params.K.to(self.device)

    def _world_to_grid_coords(self, points: torch.Tensor) -> torch.Tensor:
        """
        Converts world coordinates to normalized grid coordinates for grid_sample.
        The output range is [-1, 1].
        """
        # Normalize to [0, 1]
        normalized = (points - self.selector.bbox_min) / \
            (self.selector.bbox_max - self.selector.bbox_min)
        # Scale to [-1, 1]
        return normalized * 2 - 1

    def _score_pose_and_calculate_mask(self, pose: torch.Tensor) -> tuple[float, torch.Tensor]:
        """
        Projects self.certainty_world_coords to camera view, calculates projected 
        certainty mask (visibility), and returns a score based on visible points.

        Args:
            pose (torch.Tensor): 4x4 Cam-to-World (T_w_c) matrix.

        Returns:
            tuple: (score: float, certainty_mask: torch.Tensor [N_voxels, 1])
        """
        cam_pos_world = pose[:3, 3]
        # margin check
        is_too_small = torch.any(cam_pos_world < self.selector.bbox_min)
        is_too_large = torch.any(cam_pos_world > self.selector.bbox_max)
        if is_too_small or is_too_large:
            return -1, []

        # Occlusion check
        cam_pos_grid_coords = self._world_to_grid_coords(
            cam_pos_world.unsqueeze(0)).float()  # [1, 3]
        grid_coords_for_sample = cam_pos_grid_coords.view(1, 1, 1, 1, 3)
        occupancy_at_pos = F.grid_sample(
            self.occupancy_grid,
            grid_coords_for_sample,
            mode='bilinear',
            align_corners=True
        ).item()
        if occupancy_at_pos > 0.5:
            return -1, []

        P_c_w = torch.linalg.inv(pose)
        P_c_w = P_c_w.unsqueeze(0).to(self.K.dtype)
        P_matrix = self.K @ P_c_w[:, :3, :]  # 1x3x4 Projection Matrix

        world_points_h = torch.cat([
            self.certainty_world_coords,
            torch.ones(self.certainty_world_coords.shape[0], 1, device=self.device)
        ], dim=1).T
        projected_h = torch.bmm(P_matrix, world_points_h.unsqueeze(
            0).expand(P_matrix.shape[0], -1, -1))

        # Depths
        depths = projected_h[:, 2, :].squeeze()
        non_zero_depth_mask = (depths.abs() > 1e-6)

        # Initialize coordinates to a default large value outside bounds
        u = torch.full_like(depths, float(self.W))
        v = torch.full_like(depths, float(self.H))
        if non_zero_depth_mask.any():
            normalized_coords = \
                projected_h[:, :2, non_zero_depth_mask] / depths[non_zero_depth_mask]
            u[non_zero_depth_mask] = normalized_coords[:, 0, :]
            v[non_zero_depth_mask] = normalized_coords[:, 1, :]

        # Visibility Mask
        depth_mask = (depths > 0) & (depths < self.selector.bbox_max[2])
        in_bounds_mask = (u >= 0) & (u < self.W) & (v >= 0) & (v < self.H)

        certainty_mask = (depth_mask & in_bounds_mask).float(
        ).unsqueeze(1)  # [N_voxels, 1]

        visible_certainty_values = self.certainty_values * certainty_mask
        score = torch.sum(visible_certainty_values).item()

        return score, visible_certainty_values

    def _calculate_wiou(self, W_i: torch.Tensor, W_j: torch.Tensor):
        weighted_intersection = torch.sum(torch.min(W_i, W_j))
        weighted_union = torch.sum(torch.max(W_i, W_j))

        if weighted_union.item() < 1e-6:
            return 0.0
        return (weighted_intersection / weighted_union).item()

    def _calculate_iou(self, mask1: torch.Tensor, mask2: torch.Tensor) -> float:
        mask1_flat = mask1.flatten().bool()
        mask2_flat = mask2.flatten().bool()

        intersection = (mask1_flat & mask2_flat).sum().float()
        # union = float(N)
        union = (mask1_flat | mask2_flat).sum().float()
        score = intersection / union
        return score.item()

    def filter_poses_with_nms(
        self,
        candidate_poses: np.ndarray,
        train_poses: np.ndarray,
        num_to_select: int,
        iou_threshold: 0.8,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Filters candidate poses using scoring and Non-Maximum Suppression for diversity.
        Returns:
            np.ndarray: [num_to_select, 4, 4] of the best, diverse poses.
        """
        if len(candidate_poses) == 0:
            return torch.tensor([]), torch.tensor([]), {}

        # 2. Get Occupancy Masks for all candidates and training poses
        all_poses = torch.from_numpy(np.concatenate(
            (train_poses, candidate_poses))).to(self.device)
        N_train = len(train_poses)

        all_masks = []
        all_scores = []
        for pose in tqdm.tqdm(all_poses, desc="Evaluate pose quality"):
            score, mask = self._score_pose_and_calculate_mask(pose)
            all_masks.append(mask)
            all_scores.append(score)
        all_scores = torch.tensor(
            all_scores, dtype=torch.float32, device=self.device)

        # Split masks back into training and candidate sets
        train_masks = all_masks[:N_train]
        candidate_masks = all_masks[N_train:]
        candidate_scores = all_scores[N_train:]
        # --- NMS ---
        sorted_indices = torch.argsort(all_scores[N_train:], descending=True)
        final_indices = []
        final_scores = []
        selected_masks = train_masks.copy()
        for idx in tqdm.tqdm(sorted_indices, desc="Applying NMS"):
            if len(final_indices) >= num_to_select:
                break

            if candidate_scores[idx] < 0:  # Skip invalid poses
                continue

            current_mask = candidate_masks[idx]
            is_redundant = False

            for selected_mask in selected_masks:
                # iou = self._calculate_iou(current_mask, selected_mask)
                iou = self._calculate_wiou(current_mask, selected_mask)

                if iou > iou_threshold:
                    is_redundant = True
                    break  # Too much visual overlap with an existing pose

            if is_redundant:
                continue

            # If it passed all checks, select it
            final_indices.append(idx.item())
            selected_masks.append(current_mask)
            final_scores.append(candidate_scores[idx])

        return final_indices, final_scores, selected_masks

    def filter_poses_with_score(
        self,
        candidate_poses: np.ndarray,
        train_poses: np.ndarray,
        num_to_select: int,
        iou_threshold: 0.8,
    ) -> np.ndarray:
        """
        Filters candidate poses using scoring and Non-Maximum Suppression for diversity.
        Returns:
            np.ndarray: [num_to_select, 4, 4] of the best, diverse poses.
        """
        if len(candidate_poses) == 0:
            return torch.tensor([]), torch.tensor([]), {}

        # 2. Get Occupancy Masks for all candidates and training poses
        all_poses = torch.from_numpy(
            np.concatenate((train_poses, candidate_poses)))
        N_train = len(train_poses)

        all_masks = []
        all_scores = []
        for pose in tqdm.tqdm(all_poses, desc="Evaluate pose quality"):
            score, mask = self._score_pose_and_calculate_mask(pose)
            all_masks.append(mask)
            all_scores.append(score)
        all_scores = torch.tensor(
            all_scores, dtype=torch.float32, device=self.device)

        # Split masks back into training and candidate sets
        train_masks = all_masks[:N_train]
        candidate_masks = all_masks[N_train:]
        candidate_scores = all_scores[N_train:]

        sorted_indices = torch.argsort(all_scores[N_train:], descending=True)
        final_indices = []
        final_scores = []
        selected_masks = train_masks.copy()
        for idx in tqdm.tqdm(sorted_indices, desc="Applying NMS"):
            if len(final_indices) >= num_to_select:
                break

            if candidate_scores[idx] < 0:  # Skip invalid poses
                continue
            current_mask = candidate_masks[idx]

            # If it passed all checks, select it
            final_indices.append(idx.item())
            selected_masks.append(current_mask)
            final_scores.append(candidate_scores[idx])

        return final_indices, final_scores, selected_masks

    def get_world_coords_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Converts 3D grid indices (i, j, k) to world coordinates (x, y, z).
        Indices shape: (N_samples, 3)
        Output shape: (N_samples, 3)
        """
        res = self.selector.resolution
        L_norm = indices.float() / (res - 1)

        bbox_min = self.selector.bbox_min
        bbox_max = self.selector.bbox_max

        bbox_size = bbox_max - bbox_min
        world_coords = L_norm.to(self.device) * \
            bbox_size[None, :] + bbox_min[None, :]

        return world_coords

    def sample_lookat_points(
        self,
        num_samples: int,
        occ_threshold: float = 0.05,
        boundary_ratio: float = 0.5
    ) -> torch.Tensor:
        """
        Samples look_at points (world coordinates) based on combined certainty and occupancy grids.

        Args:
            num_samples: Number of world coordinates to sample.
            occ_threshold: Minimum occupancy score required for a voxel to be sampled.

        Returns:
            torch.Tensor: Sampled world coordinates. Shape (num_samples, 3).
        """
        # (Resolution, Resolution, Resolution)
        certainty_grid = self.certainty_grid.squeeze()
        occupancy_grid = self.occupancy_grid.squeeze()
        res = self.selector.resolution

        margin_voxels = int(res * boundary_ratio)
        start_idx = margin_voxels
        end_idx = res - margin_voxels

        # 2. initial mask and only set central area = 1
        space_mask = torch.zeros(
            (res, res, res), device=self.device, dtype=torch.float32)
        if start_idx < end_idx:
            space_mask[start_idx:end_idx,
                       start_idx:end_idx, start_idx:end_idx] = 1.0

        occ_mask = (occupancy_grid > occ_threshold).float()
        weights = certainty_grid * occ_mask * space_mask

        pmf_flat = weights.view(-1)
        valid_indices = torch.nonzero(pmf_flat).squeeze(-1)
        valid_pmf = pmf_flat[valid_indices]

        if valid_pmf.sum() == 0:
            print(
                "Warning: No valid points in the central region. Falling back to global random sampling.")
            return []

        sampled_idx_flat = torch.multinomial(
            valid_pmf, num_samples, replacement=True)
        original_idx_flat = valid_indices[sampled_idx_flat]

        k = original_idx_flat % res
        j = (original_idx_flat // res) % res
        i = original_idx_flat // (res * res)

        # (N_samples, 3)
        indices_3d = torch.stack([i, j, k], dim=-1).long()
        lookat_points = self.get_world_coords_from_indices(indices_3d)

        return lookat_points


class CameraPoseVisualizer:
    def __init__(self, xlim, ylim, zlim):
        self.fig = plt.figure(figsize=(10, 6))
        self.ax = self.fig.add_subplot(projection='3d')
        self.ax.set_aspect("auto")
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.set_zlim(zlim)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        plt.title('Camera Extrinsics Visualization')

    def extrinsic2pyramid(
        self,
        extrinsic,
        color_value=0.5,
        hw_ratio=0.75,
        base_xval=0.1,
        zval=0.3
    ):
        """
        Draw a small camera frustum/pyramid to represent the extrinsic matrix.
        extrinsic : (4,4) camera-to-world transform
        color_value : float in [0,1], mapped to a color via colormap
        hw_ratio : The aspect ratio of the camera plane
        base_xval, zval : size/length scalars for the frustum drawing
        """
        vertex_std = np.array([
            [0, 0, 0, 1],
            [base_xval, -base_xval * hw_ratio, zval, 1],
            [base_xval,  base_xval * hw_ratio, zval, 1],
            [-base_xval,  base_xval * hw_ratio, zval, 1],
            [-base_xval, -base_xval * hw_ratio, zval, 1]
        ])
        # Transform these points by the given extrinsic (camera-to-world).
        vertex_transformed = vertex_std @ extrinsic.T

        # Create triangular faces for the frustum
        meshes = [
            [vertex_transformed[0, :-1], vertex_transformed[1, :-1],
                vertex_transformed[2, :-1]],
            [vertex_transformed[0, :-1], vertex_transformed[2, :-1],
                vertex_transformed[3, :-1]],
            [vertex_transformed[0, :-1], vertex_transformed[3, :-1],
                vertex_transformed[4, :-1]],
            [vertex_transformed[0, :-1], vertex_transformed[4, :-1],
                vertex_transformed[1, :-1]],
            [vertex_transformed[1, :-1], vertex_transformed[2, :-1],
                vertex_transformed[3, :-1], vertex_transformed[4, :-1]]
        ]

        color = plt.cm.rainbow(color_value)
        self.ax.add_collection3d(
            Poly3DCollection(meshes, facecolors=color,
                             linewidths=0.5, edgecolors=color, alpha=0.4)
        )

    def colorbar(self, max_value):
        cmap = mpl.cm.rainbow
        norm = mpl.colors.Normalize(vmin=0, vmax=max_value)
        self.fig.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            ax=self.ax,
            orientation='vertical',
            label='Frame Number'
        )

    def show(self, save_path=None):
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, format='jpg', dpi=300)
        # plt.show()


def extrinsic2pyramid(
    ax,
    extrinsic,
    color_value=0.5,
    hw_ratio=0.75,
    base_xval=0.1,
    zval=0.3
):
    """
    Draw a small camera frustum/pyramid to represent the extrinsic matrix.
    extrinsic : (4,4) camera-to-world transform
    color_value : float in [0,1], mapped to a color via colormap
    hw_ratio : The aspect ratio of the camera plane
    base_xval, zval : size/length scalars for the frustum drawing
    """
    vertex_std = np.array([
        [0, 0, 0, 1],
        [base_xval, -base_xval * hw_ratio, zval, 1],
        [base_xval,  base_xval * hw_ratio, zval, 1],
        [-base_xval,  base_xval * hw_ratio, zval, 1],
        [-base_xval, -base_xval * hw_ratio, zval, 1]
    ])
    # Transform these points by the given extrinsic (camera-to-world).
    vertex_transformed = vertex_std @ extrinsic.T

    # Create triangular faces for the frustum
    meshes = [
        [vertex_transformed[0, :-1], vertex_transformed[1, :-1],
            vertex_transformed[2, :-1]],
        [vertex_transformed[0, :-1], vertex_transformed[2, :-1],
            vertex_transformed[3, :-1]],
        [vertex_transformed[0, :-1], vertex_transformed[3, :-1],
            vertex_transformed[4, :-1]],
        [vertex_transformed[0, :-1], vertex_transformed[4, :-1],
            vertex_transformed[1, :-1]],
        [vertex_transformed[1, :-1], vertex_transformed[2, :-1],
            vertex_transformed[3, :-1], vertex_transformed[4, :-1]]
    ]

    color = plt.cm.rainbow(color_value)
    ax.add_collection3d(
        Poly3DCollection(
            meshes, facecolors=color, linewidths=0.5, edgecolors=color, alpha=0.4)
    )
