import copy
import os
import random
from pathlib import Path
import time
import json
from typing import List, Literal

from brisque import BRISQUE
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import torch
import torch.nn.functional as F
import imageio.v3 as iio
from omegaconf import OmegaConf
from PIL import Image
from sklearn.cluster import KMeans

from conerf.evaluators.gaussian_splatting_evaluator import GaussianSplatEvaluator
from conerf.gaussian_fields.app_embed import AppearanceOptModule
from conerf.utils.utils import save_sampled_images, get_subdirs, colorize
from conerf.datasets.utils import create_dataset
from conerf.base.checkpoint_manager import CheckPointManager
from conerf.datasets.dataset_base import compose_cameras, compose_camera
from conerf.datasets.camera_traj import create_preset_poses, get_lookat, get_lookat2
from conerf.utils.trajs_utils import ViewSelection, PoseFilter, CameraPoseVisualizer
from conerf.visualization.viser_gui import (
    GaussianSplattingVisualizer, compute_rainbow_color
)

from difix3d.pipeline_difix import DifixPipeline
from difix3d.utils import CameraPoseInterpolator


def visualize_cameras(
    trajectory,
    path: str,
    ref=None,
    save_name='camera_trajectories',
    legend_name=None
):
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122)

    # Make trajectory always (Nseq, Ncams, 4, 4)
    if not isinstance(trajectory, list) and trajectory.ndim == 3:
        trajectory = trajectory[np.newaxis, ...]

    n_traj = len(trajectory)
    colors = plt.cm.viridis(np.linspace(0, 1, n_traj))
    for ti, traj in enumerate(trajectory):
        positions = traj[:, :3, 3]
        forwards = traj[:, :3, 2]
        color = colors[ti]

        label = f'Trajectory {ti+1}' if legend_name is None else legend_name[ti]
        ax1.scatter(positions[:, 0], positions[:, 1],
                    positions[:, 2], color=color, s=50, label=label)
        ax2.scatter(positions[:, 0], positions[:, 1],
                    color=color, s=8, label=label)

        ax1.quiver(positions[:, 0], positions[:, 1], positions[:, 2],
                   forwards[:, 0], forwards[:, 1], forwards[:, 2],
                   length=0.2, normalize=True, color=color)

    if ref is not None:
        colors = plt.cm.viridis(np.linspace(0, 1, len(ref)))
        ref = np.asarray(ref)
        pos_ref = ref[:, :3, 3]
        f_ref = ref[:, :3, 2]
        ax1.scatter(pos_ref[:, 0], pos_ref[:, 1],
                    pos_ref[:, 2], color=colors, s=200)
        ax1.quiver(pos_ref[:, 0], pos_ref[:, 1], pos_ref[:, 2],
                   f_ref[:, 0], f_ref[:, 1], f_ref[:, 2],
                   length=0.4, normalize=True, color='red')
        ax2.scatter(pos_ref[:, 0], pos_ref[:, 1], color=colors, s=50)

    ax1.set_title('3D Camera Trajectories')
    ax1.set_xlabel('X-Axis')
    ax1.set_ylabel('Y-Axis')
    ax1.set_zlabel('Z-Axis')
    ax1.view_init(elev=20, azim=30)
    ax2.set_title('Top View of Camera Trajectories')
    ax2.set_xlabel('X-Axis')
    ax2.set_ylabel('Y-Axis')
    ax2.axis('equal')

    ax1.legend()
    ax2.legend()
    plt.tight_layout()
    plt.savefig(f'{path}/{save_name}.png')
    plt.close(fig)


def plot_traj(camtoworlds, save_path, index, normalize=True, ref_c2w=None):
    if normalize:
        centers = [mat[:3, 3] for mat in camtoworlds]
        centers = np.array(centers)
        mean_center = np.mean(centers, axis=0)
        centers -= mean_center
        max_dist = np.max(np.linalg.norm(centers, axis=1))
        if max_dist > 1e-9:
            centers /= max_dist
        # Put back the adjusted translations
        for i, mat in enumerate(camtoworlds):
            camtoworlds[i][:3, 3] = centers[i]

    viz = CameraPoseVisualizer([-1, 1], [-1, 1], [-1, 1])
    max_idx = max(len(camtoworlds) - 1, 1)
    for i, c2w in enumerate(camtoworlds):
        color_val = i / float(max_idx)
        viz.extrinsic2pyramid(
            c2w, color_val, hw_ratio=9.0 / 16.0, base_xval=0.1, zval=0.3)

    if ref_c2w is not None:
        for i, c2w in enumerate(ref_c2w):
            color_val = i / float(max_idx)
            viz.extrinsic2pyramid(
                c2w, 1.0, hw_ratio=9.0 / 16.0, base_xval=0.1, zval=0.3)

    # 5) Add colorbar and show/save
    viz.colorbar(len(camtoworlds))
    viz.show(save_path + '/' + str(index))


def get_closest_camera_per_cluster(
    camera_positions: np.ndarray,  # [N, 3]
    labels: np.ndarray,            # [N,]
    cluster_centers: np.ndarray    # [k, 3]
) -> np.ndarray:

    closest_indices = []
    for cluster_id in range(len(cluster_centers)):
        cluster_mask = (labels == cluster_id)
        cluster_positions = camera_positions[cluster_mask]
        cluster_indices = np.where(cluster_mask)[0]
        if len(cluster_positions) == 0:
            closest_indices.append(-1)
            continue

        distances = np.linalg.norm(
            cluster_positions - cluster_centers[cluster_id], axis=1)
        closest_idx = cluster_indices[np.argmin(distances)]
        closest_indices.append(closest_idx)
    return np.array(closest_indices)


def cluster_cameras(c2w, num_clusters=5, num_selection=5):
    """
    Clusters cameras based on their positions in 3D space.
    Returns a tensor of shape (N, 3) where N is the number of clusters.
    """
    positions = c2w[:, :3, 3]
    # Use k-means clustering to find clusters in camera positions
    # Limit to a maximum of 10 clusters
    n_clusters = min(num_clusters, len(positions))
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(positions)
    # return torch.tensor(kmeans.cluster_centers_, dtype=c2w.dtype)
    cluster_centers = kmeans.cluster_centers_  # [k, 3]
    labels = kmeans.labels_  # [N,]
    cluster_indices = get_closest_camera_per_cluster(
        positions, labels, cluster_centers)

    num_from_clusters = min(len(cluster_indices),
                            random.choice(range(num_selection)))
    num_from_random = num_selection - num_from_clusters

    selected_cluster_indices = np.random.choice(
        cluster_indices, size=num_from_clusters, replace=False)
    all_indices = np.arange(len(c2w))
    remain_pool_indices = np.setdiff1d(all_indices, selected_cluster_indices)
    randomly_selected_indices = np.random.choice(
        remain_pool_indices, size=num_from_random, replace=False)

    final_indices = np.concatenate(
        [selected_cluster_indices, randomly_selected_indices])
    return c2w[final_indices]


def fps_camera_selection(c2w, num_anchors=10):
    positions = c2w[:, :3, 3]

    N = len(positions)
    anchors = np.zeros((num_anchors,), dtype=np.int32)
    anchors[0] = np.random.randint(N)
    min_distances = np.full(shape=(N,), fill_value=np.inf)
    for i in range(1, num_anchors):
        last_anchor = positions[anchors[i-1]]
        dists = np.linalg.norm(positions - last_anchor, axis=1)
        min_distances = np.minimum(min_distances, dists)
        anchors[i] = np.argmax(min_distances)
    return c2w[anchors]


def jitter_camera_pose(c2w, jitter_ratio=0.1, position_std=0.1, rotation_deg=10.0):
    B = c2w.shape[0]
    jitter_num = int(B * jitter_ratio)
    c2w_jittered = c2w
    idx = np.random.randint(0, B, size=jitter_num)
    # ---- 1. position jitter ----
    pos_jitter = np.random.randn(jitter_num, 3) * position_std
    c2w_jittered[idx, :3, 3] += pos_jitter
    # ---- 2. rotation jitter ----
    angle = (np.random.rand(jitter_num, 1) - 0.5) * 2 * \
        rotation_deg * np.pi / 180.0  # [-deg, deg]
    axis = np.random.randn(jitter_num, 3)
    axis = axis / np.linalg.norm(axis, axis=-1, keepdims=True)
    K = np.zeros((jitter_num, 3, 3))
    K[:, 0, 1] = -axis[:, 2]
    K[:, 0, 2] = axis[:, 1]
    K[:, 1, 0] = axis[:, 2]
    K[:, 1, 2] = -axis[:, 0]
    K[:, 2, 0] = -axis[:, 1]
    K[:, 2, 1] = axis[:, 0]

    I = np.tile(np.eye(3), (jitter_num, 1, 1))
    K2 = np.matmul(K, K)
    R = I + np.sin(angle)[..., None] * K + \
        (1-np.cos(angle)[..., None]) * K2   # Rodrigues公式
    old_rot = c2w_jittered[idx, :3, :3]
    new_rot = np.einsum('bij,bjk->bik', R, old_rot)
    c2w_jittered[idx, :3, :3] = new_rot
    return c2w_jittered


def load_dataset(config: OmegaConf, device: str = 'cuda', split="train"):
    val_config = copy.deepcopy(config)
    val_config.dataset.multi_blocks = False
    val_config.dataset.num_blocks = 1
    if split == "train":
        dataset = create_dataset(
            config=val_config,
            split=val_config.dataset.train_split,
            num_rays=None,
            apply_mask=val_config.dataset.apply_mask,
            device=device
        )
    elif split == "val":
        dataset = create_dataset(
            config=val_config,
            split=val_config.dataset.val_split,
            num_rays=None,
            apply_mask=val_config.dataset.apply_mask,
            device=device
        )
    else:
        print(f"Don't support {split}")
        exit()

    return dataset


def calculate_wiou_list_batched(
    W_masks: list[torch.Tensor],
    batch_size: int = 64
) -> torch.Tensor:
    """
    Calculates a pairwise IoU matrix for a list of masks, robust to mask dimensions.
    """
    N = len(W_masks)
    if N == 0:
        return torch.empty((0, 0), dtype=torch.float32)

    all_W = torch.stack(W_masks, dim=0).float()
    all_W = all_W.flatten(start_dim=1)  # Shape: [N, L]

    device = all_W.device
    wiou_matrix = torch.zeros((N, N), device=device, dtype=torch.float32)
    sum_W = all_W.sum(dim=1)

    for i in range(0, N, batch_size):
        i_end = min(i + batch_size, N)
        W_A = all_W[i:i_end]  # Shape: [B, L]
        W_B = all_W           # Shape: [N, L]

        # Use efficient matrix multiplication to calculate intersections
        weighted_intersection_batch = torch.min(
            W_A[:, None, :], W_B[None, :, :]).sum(dim=2)  # Shape: [B, N]
        sum_A_batch = sum_W[i:i_end]  # Shape: [B]
        sum_B = sum_W
        weighted_union_batch = sum_A_batch.unsqueeze(
            1) + sum_B.unsqueeze(0) - weighted_intersection_batch  # Shape: [B, N]

        wiou_batch = weighted_intersection_batch / \
            (weighted_union_batch + 1e-6)  # Shape: [B, N]
        wiou_matrix[i:i_end, :] = wiou_batch

    return wiou_matrix


def calculate_iou_list_batched(
    masks: list[torch.Tensor],
    batch_size: int = 64
) -> torch.Tensor:
    """
    Calculates a pairwise IoU matrix for a list of masks, robust to mask dimensions.
    """
    N = len(masks)
    if N == 0:
        return torch.empty((0, 0), dtype=torch.float32)

    all_masks = torch.stack(masks, dim=0).float()  # Shape: [N, H, W, ...]
    all_masks = all_masks.flatten(start_dim=1)      # Shape: [N, L]

    device = all_masks.device
    iou_matrix = torch.zeros((N, N), device=device, dtype=torch.float32)

    for i in range(0, N, batch_size):
        i_end = min(i + batch_size, N)
        masks_A = all_masks[i:i_end]  # Shape: [B, L]
        masks_B = all_masks           # Shape: [N, L]

        # Use efficient matrix multiplication to calculate intersections
        intersection_batch = torch.matmul(masks_A, masks_B.T)  # Shape: [B, N]

        # Calculate union from sums and intersection
        sum_A = masks_A.sum(dim=1)  # Shape: [B]
        sum_B = masks_B.sum(dim=1)  # Shape: [N]
        union_batch = sum_A.unsqueeze(
            1) + sum_B.unsqueeze(0) - intersection_batch  # Shape: [B, N]

        # Add a small epsilon to avoid division by zero
        iou_batch = intersection_batch / (union_batch + 1e-6)  # Shape: [B, N]
        iou_matrix[i:i_end, :] = iou_batch

    return iou_matrix


def build_view_graph(node_ids, iou_matrix: torch.Tensor, top_k: int) -> dict:
    N = iou_matrix.shape[0]
    view_graph = {str(node_ids[i]): [] for i in range(N)}
    if N == 0:
        return view_graph
    iou_matrix = iou_matrix.fill_diagonal_(0.0)
    top_k_values, top_k_indices = torch.topk(
        iou_matrix, k=min(top_k, N), dim=1)

    for i in range(N):
        cur_id = node_ids[i]
        for k in range(min(top_k, N)):
            n_idx = top_k_indices[i, k].item()
            neighbor_id = node_ids[n_idx]
            weight = top_k_values[i, k].item()
            if weight > 1e-4:
                view_graph[str(cur_id)].append((neighbor_id, weight))
    return view_graph


class CameraSampler(GaussianSplatEvaluator):
    """Class for evaluating Gaussian Splatting models."""

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
        self.config = config
        self.device = device
        self.verbose = verbose

        self.models = models
        self.model_iterations = list()
        self.meta_data = meta_data

        self.output_dir = os.path.join(
            self.config.dataset.output_dir, config.expname)
        os.makedirs(self.output_dir, exist_ok=True)

        self.eval_dir = os.path.join(self.output_dir, "renders")
        os.makedirs(self.eval_dir, exist_ok=True)

        self.app_module = None

        if config.appearance.use_app_embed:
            feature_dim = config.appearance.app_feat_dim
            embed_dim = config.appearance.app_embed_dim
            self.app_module = AppearanceOptModule(
                0,
                feature_dim,
                embed_dim,
                config.texture.max_sh_degree,
            ).to(device)

        # only load trainset
        train_dataset = load_dataset(config, device='cpu')
        self.train_cameras = train_dataset.cameras
        if self.config.dataset.val_interval > 0:
            val_dataset = load_dataset(config, split="val", device='cpu')
            self.val_cameras = val_dataset.cameras

        self.color_bkgd = torch.tensor(
            [0, 0, 0], dtype=torch.float32, device=self.device)

        if self.app_module is not None and self.app_module.embeds.weight.shape[0] == 0:
            embed_dim = self.config.appearance.app_embed_dim
            self.app_module.embeds = torch.nn.Embedding(
                self.config.appearance.input_dim, embed_dim).to(self.device)

        if models is None:
            self.load_model()
        self.global_model = None

        # cameras clustering & visualization
        view_selector = ViewSelection(self.models[0])
        self.pose_filter = PoseFilter(
            view_selector, camera_params=self.train_cameras[0])
        self.lookup_pool = self.pose_filter.sample_lookat_points(
            self.config.sampler.num_lookup, occ_threshold=0.5)
        self.traj_modes = self.config.sampler.traj_modes
        self.image_checker = BRISQUE(url=False)

        # initial DIFIX
        self.interpolator = CameraPoseInterpolator(
            rotation_weight=1.0, translation_weight=1.0)
        self.difix = DifixPipeline.from_pretrained(
            "nvidia/difix_ref", trust_remote_code=True)
        self.difix.set_progress_bar_config(disable=True)
        self.difix.to("cuda")

    def setup_visualizer(self):
        """
        Initialize the Rerun visualizer with COLMAP data.

        Args:
            colmap_data: Dictionary from your load_colmap method containing:
                - 'images': RGB images (N, 3, H, W)
                - 'poses': Camera poses (N, 4, 4) in camera-to-world
                - 'intrinsics': Camera intrinsics (N, 3, 3)
        """
        scene_dir = os.path.join(
            self.config.dataset.root_dir, self.config.dataset.scene)
        colmap_dir = os.path.join(
            scene_dir, "sparse",
            "manhattan_world" if self.config.dataset.use_manhattan_world else "0"
        )
        image_dir = os.path.join(scene_dir, "images")
        if self.config.dataset.factor > 1:
            image_dir = image_dir + f"_{self.config.dataset.factor}"

        self.visualizer = GaussianSplattingVisualizer(
            colmap_path=Path(colmap_dir), images_path=Path(image_dir),
            splats_path=Path("placeholder.splat"),
            cameras_to_reserve=100,
        )

    def load_model(self):
        self.meta_data, self.models = [], []  # pylint: disable=W0201
        ckpt_manager = CheckPointManager(verbose=False)

        input_model_dir = os.path.join(
            self.config.dataset.load_from, self.config.expname)

        assert os.path.exists(input_model_dir), \
            f"input model directory does not exist: {input_model_dir}"
        if self.config.dataset.multi_blocks:
            model_dirs = get_subdirs(input_model_dir, "block_")
        else:
            model_dirs = [input_model_dir]

        pbar = tqdm.trange(len(model_dirs), desc="Loading Models", leave=True)
        for model_dir in model_dirs:
            local_config = copy.deepcopy(self.config.trainer)
            local_config.ckpt_path = os.path.join(model_dir, 'model.pth')
            assert os.path.exists(local_config.ckpt_path), \
                f"checkpoint does not exist: {local_config.ckpt_path}"

            with open(
                os.path.join(model_dir, "meta_non_splats.json"), "r", encoding="utf-8"
            ) as f:
                non_splat_meta_data = json.load(f)

            model = self._build_networks()
            model.active_sh_degree = non_splat_meta_data["active_sh_degree"]

            print(f'Loading model from {model_dir}')
            if self.config.compression.get("enabled", False):
                model.decompress(model_dir)
            else:
                model.load_ply(os.path.join(model_dir, 'final_splat.ply'))

            iteration = non_splat_meta_data["iteration"]
            self.meta_data.append(non_splat_meta_data)
            self.model_iterations.append(iteration)

            if self.config.appearance.use_app_embed:
                embed_dim = self.config.appearance.app_embed_dim
                # Reset appearance embedding to have length equal to training views
                # for loading checkpoints, though the we don't need appearance embedding
                # from training views during evaluation.
                self.app_module.embeds = torch.nn.Embedding(
                    self.config.appearance.input_dim, embed_dim
                ).to(self.device)
                ckpt_manager.load(local_config, models={
                                  "app_module": self.app_module})

            self.models.append(model)

            pbar.update(1)

    def select_lookat_points(
        self,
        anchors_th: torch.Tensor,
        traj_modes: str,
        p_sample_pool: float = 0.5,
    ) -> torch.Tensor:
        B = anchors_th.shape[0]
        device = anchors_th.device

        anchor_positions = anchors_th[:, :3, 3]
        if len(self.lookup_pool) < B:
            sample_from_pool_mask = torch.full((B,), False, device=device)
        else:
            sample_from_pool_mask = (torch.rand(
                B, device=device) < p_sample_pool)
        look_at = torch.empty_like(anchor_positions)

        if sample_from_pool_mask.any():
            pool_anchors = anchor_positions[sample_from_pool_mask]  # B_pool, 3

            distances_sq = torch.sum(
                (pool_anchors.unsqueeze(1) - self.lookup_pool.unsqueeze(0)) ** 2,
                dim=-1
            )

            closest_indices = torch.argmin(distances_sq, dim=1)
            sampled_look_at = self.lookup_pool[closest_indices]
            look_at[sample_from_pool_mask] = sampled_look_at

        default_mask = ~sample_from_pool_mask
        if default_mask.any():
            default_anchors = anchors_th[default_mask]  # B_default, 4, 4

            if traj_modes in [
                "move-up", "move-down", "move-left", "move-right",
                "move-forward", "move-backward",
                "dollyzoom-in", "dollyzoom-out"
            ]:
                default_look_at = get_lookat2(default_anchors, distance=2)

            elif traj_modes in ["orbit", "spiral", "lemniscate"]:
                default_look_at = get_lookat(
                    default_anchors[:, :3, 3],
                    default_anchors[:, :3, 2],
                )
            look_at[default_mask] = default_look_at

        return look_at

    def eval(
        self,
        iteration: int = None,
        split: Literal["val", "test"] = "val",
    ) -> dict:
        """
        Main logic for evaluation.
        """
        time_start = time.time()
        metrics = dict()

        val_dir = self.eval_dir
        os.makedirs(val_dir, exist_ok=True)

        camtoworlds = torch.stack(
            [cam.cam_to_world for cam in self.train_cameras])
        camtoworlds = camtoworlds.cpu().numpy()
        N_data = len(camtoworlds)
        if self.config.dataset.val_interval > 0:
            val_c2ws = torch.stack(
                [cam.cam_to_world for cam in self.val_cameras])
            val_c2ws = val_c2ws.cpu().numpy()
        else:
            val_c2ws = camtoworlds

        for k, model in enumerate(self.models):
            if model.get_xyz.device == torch.device('cpu'):
                model = model.to("cuda")
            else:
                model.eval()

            sampled_c2ws = []
            sampled_Ks = []
            for mode in self.traj_modes:
                c2ws, Ks = self.sample_trajectory(
                    camtoworlds, self.train_cameras[0].K, mode,
                    num_frames=self.config.sampler.num_frames
                )
                sampled_c2ws.extend(c2ws)
                sampled_Ks.extend(Ks)

            sampled_c2ws = jitter_camera_pose(
                np.array(sampled_c2ws),
                jitter_ratio=0.5,
                position_std=random.uniform(0, 0.5),
                rotation_deg=random.uniform(0, 30),
            )
            sampled_Ks = np.array(sampled_Ks)

            # poses scoring and matching
            indexs, scores, surface_masks = self.pose_filter.filter_poses_with_nms(
                sampled_c2ws, camtoworlds, num_to_select=self.config.sampler.nms_topN,
                iou_threshold=self.config.sampler.nms_iou,
            )
            selected_c2ws = sampled_c2ws[indexs]
            selected_Ks = sampled_Ks[indexs]

            selected_c2ws = torch.from_numpy(
                selected_c2ws).float().to(self.device)
            selected_Ks = torch.from_numpy(selected_Ks).float().to(self.device)

            image_paths = [i for i in range(len(selected_c2ws))]
            cameras = compose_cameras(
                image_paths,
                None,
                selected_c2ws,
                selected_Ks,
                channels=self.config.dataset.get("num_channels", 3),
                device='cpu',
            )

            # render
            aligned_cameras = []
            valid_index = 0
            confidences = []
            node_masks = copy.deepcopy(surface_masks[:N_data])
            node_names = list(range(N_data))

            selected_masks = []
            sampled_trajs = []
            revised_trajs = []
            prev_trajs = []
            for i in tqdm.trange(len(cameras), desc="Rendering trajectory"):
                camera = cameras[i]
                camera = camera.copy_to_device(self.device)
                img_name = f"{valid_index:03d}"
                depth_score, rgb_score, success = self._eval(
                    camera, model, self.eval_dir, img_name, threshold=0.5)

                num_try = 0
                if random.random() > 0.5:
                    revised_refs = camtoworlds
                else:
                    revised_refs = val_c2ws
                cur_c2w = camera.cam_to_world.cpu().numpy()
                while not success and num_try < len(self.config.sampler.interpolate_dist):
                    new_pose, train_indx = self.interpolator.shift_poses(
                        revised_refs, [cur_c2w],
                        distance=self.config.sampler.interpolate_dist[num_try]
                    )
                    # new camera
                    new_camera = compose_camera(
                        image_index=f"fv_{valid_index}",
                        image_path=os.path.join(
                            self.eval_dir, f"rgb", f"{img_name}.png"),
                        image=None,
                        camtoworld=torch.from_numpy(new_pose[0]).to(
                            camera.cam_to_world.dtype),
                        intrinsics=camera.K[0],
                        depth_path=os.path.join(
                            self.eval_dir, "depth", f"{img_name}.png"),
                        mask_path=None,
                        normal=None,
                    )
                    depth_score, rgb_score, success = self._eval(
                        new_camera, model, self.eval_dir, img_name,
                        threshold=self.config.sampler.max_brisque
                    )
                    num_try += 1

                if success:
                    if num_try == 0:
                        camera.image_index = f"fv_{valid_index}"
                        camera.image_path = os.path.join(
                            self.eval_dir, f"rgb", f"{img_name}.png")
                        camera.depth_path = os.path.join(
                            self.eval_dir, "depth", f"{img_name}.png")
                        aligned_cameras.append(camera)
                    else:
                        aligned_cameras.append(new_camera)
                    confidences.append(
                        [scores[i].detach().cpu().numpy(), depth_score, rgb_score]
                    )
                    selected_masks.append(surface_masks[N_data + i])
                    valid_index += 1
                    # visualize
                    if num_try == 0:
                        sampled_trajs.append(cur_c2w)
                    else:
                        prev_trajs.append(cur_c2w)
                        revised_trajs.append(new_pose[0])
            # visualization
            if len(prev_trajs) > 0:
                total_trajs = [
                    camtoworlds, val_c2ws, np.stack(
                    prev_trajs), np.stack(sampled_trajs + revised_trajs)
                ]
                visualize_cameras(
                    total_trajs, self.eval_dir,
                    save_name="val_vs_sampled",
                    legend_name=["train", "val", "before rectification", "freeview"]
                )

            # v2
            if success:
                total_trajs = [camtoworlds, np.stack(
                    sampled_trajs + revised_trajs)]
                visualize_cameras(total_trajs, self.eval_dir, save_name="val_vs_fv", legend_name=[
                                  "train", "freeview"])

            # filter images to meet numbers
            confidences = np.array(confidences)
            (
                final_aligned_cameras, final_confidences, final_node_names, final_selected_masks
            ) = dynamic_filtering(
                confidences, aligned_cameras, selected_masks,
                target_count=100,
                initial_threshold=0.25,
                max_threshold=self.config.sampler.max_brisque,
            )
            assert (
                len(final_aligned_cameras) == len(final_confidences) and 
                len(final_aligned_cameras) == len(final_selected_masks)
            ), f"Error length cameras: {len(final_aligned_cameras)}, " + \
               f"confidences: {len(final_confidences)}, masks: {len(final_selected_masks)}"
            print("Finish filtering images.")

            # save camera
            if len(final_aligned_cameras) > 0:
                torch.save(final_aligned_cameras, os.path.join(
                    self.eval_dir, f"cameras.pt"))
                with open(f"{self.eval_dir}/conf.txt", 'w') as f:
                    for item in final_confidences:
                        f.write(" ".join(item) + '\n')

                # get and save final view graph
                node_names.extend(final_node_names)
                node_masks.extend(final_selected_masks)
                iou_matrix = calculate_wiou_list_batched(
                    node_masks, batch_size=16)
                view_graph = build_view_graph(
                    node_names, iou_matrix, top_k=self.config.sampler.graph_topN)
                with open(os.path.join(self.eval_dir, f"view_graph.json"), 'w') as f:
                    json.dump(view_graph, f)

                time_end = time.time()

                view_graph = json.load(
                    open(os.path.join(self.eval_dir, f"view_graph.json"), "r"))
                self.run_difix(view_graph)
            else:
                time_end = -1

        training_time = time_end - time_start
        stats = {
            "Training Time": training_time,
        }
        metric_file = os.path.join(self.eval_dir, 'stats.json')
        json_obj = json.dumps(stats, indent=4)
        with open(metric_file, 'w', encoding='utf-8') as file:
            file.write(json_obj)
        return metrics

    @torch.no_grad()
    def sample_trajectory(
        self,
        training_c2ws,
        intrinsics,
        traj_modes='spiral',
        num_frames=60
    ):
        if traj_modes == "interp":
            sampled_c2ws_list = []
            sampled_Ks = []
            for _ in range(self.config.sampler.pre_mode_trajs):
                ref_cameras = cluster_cameras(
                    training_c2ws,
                    num_clusters=5,
                    num_selection=random.randint(3, 6),
                ).astype(np.float32)

                jitter_refs = jitter_camera_pose(
                    ref_cameras,
                    jitter_ratio=1,
                    position_std=0.1,
                    rotation_deg=random.uniform(0, 20)
                )
                # sample preset poses
                sampled_c2ws, sampled_K = create_preset_poses(
                    traj_modes, torch.from_numpy(jitter_refs), None, None, intrinsics,
                    n_steps=num_frames // 2,
                    zoom_factor=None
                )  # c2ws:[n_frames, 4, 4] Ks:[n_frames, 4, 4]
                sampled_c2ws_list.append(sampled_c2ws)
                sampled_Ks.append(sampled_K)
            sampled_Ks = np.concatenate(sampled_Ks, axis=0)
        else:
            ref_cameras = cluster_cameras(
                training_c2ws,
                num_clusters=5,
                num_selection=self.config.sampler.pre_mode_trajs
            ).astype(np.float32)

            jitter_refs = jitter_camera_pose(
                ref_cameras,
                jitter_ratio=1,
                position_std=0.1,
                rotation_deg=random.uniform(0, 20)
            )
            # visualize_cameras(training_c2ws, self.eval_dir, ref_cameras, save_name='fps')
            anchors_th = torch.as_tensor(ref_cameras)

            look_at = self.select_lookat_points(
                anchors_th, traj_modes, p_sample_pool=0.5)
            # average up
            up = - \
                F.normalize(torch.from_numpy(
                    ref_cameras[:, :3, 1]).mean(0), dim=-1)

            # sample preset poses
            sampled_c2ws_list, sampled_Ks = create_preset_poses(
                traj_modes, torch.from_numpy(jitter_refs), look_at, up, intrinsics,
                n_steps=num_frames,
                zoom_factor=None
            )  # c2ws:[B, n_frames, 4, 4] Ks:[n_frames, 4, 4]

        visualize_cameras(
            sampled_c2ws_list, self.eval_dir,
            ref_cameras, save_name=traj_modes
        )
        sampled_c2ws_list = np.concatenate(sampled_c2ws_list, axis=0)

        return sampled_c2ws_list, sampled_Ks

    @torch.no_grad()
    def _eval(self, data, model, eval_dir, image_name, traj_mode="", threshold=0):
        from conerf.render.gaussian_render import render_gsplat
        torch.cuda.reset_peak_memory_stats()

        image_dict = {}
        precompute_colors = None
        if self.app_module is not None:
            dirs = model.get_xyz - data.camera_center.repeat(
                model.get_features.shape[0], 1)
            precompute_colors = self.app_module(
                features=model.get_features,
                embed_ids=None,
                dirs=dirs[None],
                sh_degree=model.max_sh_degree,
                embed_value=0,
            )
            precompute_colors = precompute_colors + model.get_colors
            precompute_colors = torch.sigmoid(precompute_colors)

        render_results = render_gsplat(
            gaussian_splat_model=model,
            viewpoint_camera=data,
            pipeline_config=self.config.pipeline,
            bkgd_color=self.color_bkgd,
            anti_aliasing=self.config.texture.anti_aliasing,
            separate_sh=False,  # True,
            override_color=precompute_colors,
        )
        colors, depth = render_results["rendered_image"], render_results["depth"]

        imgs = colors.permute(1, 2, 0).cpu().numpy()
        # imgs = (imgs * 255).astype(np.uint8)
        depth_score, rgb_score = self.check_render_quality(imgs, depth)

        if depth_score < 0 or rgb_score < 0 or rgb_score > threshold:
            return depth_score, rgb_score, False

        colors = torch.clamp(colors.cpu(), 0, 1)
        depth = colorize(depth.cpu().squeeze(0), cmap_name="jet")
        image_dict[f"rgb"] = colors.permute(1, 2, 0)
        image_dict["depth"] = depth
        save_sampled_images(
            save_dir=eval_dir, image_dict=image_dict, name=image_name, suffix=traj_mode)

        # render appearance variants
        image_dict = {}
        if self.app_module is not None:
            # for app_i, value in enumerate([-4, -2, 2, 4, 6]):
            for app_i, value in enumerate([-4, 6]):
                dirs = model.get_xyz - data.camera_center.repeat(
                    model.get_features.shape[0], 1)
                precompute_colors = self.app_module(
                    features=model.get_features,
                    embed_ids=None,
                    dirs=dirs[None],
                    sh_degree=model.max_sh_degree,
                    embed_value=value
                )
                precompute_colors = precompute_colors + model.get_colors
                precompute_colors = torch.sigmoid(precompute_colors)

                render_results = render_gsplat(
                    gaussian_splat_model=model,
                    viewpoint_camera=data,
                    pipeline_config=self.config.pipeline,
                    bkgd_color=self.color_bkgd,
                    anti_aliasing=self.config.texture.anti_aliasing,
                    separate_sh=False,  # True,
                    override_color=precompute_colors,
                )
                colors = render_results["rendered_image"]
                colors = torch.clamp(colors.cpu(), 0, 1)
                image_dict[f"rgb_{app_i + 1}"] = colors.permute(1, 2, 0)

            save_sampled_images(
                save_dir=eval_dir, image_dict=image_dict, name=image_name, suffix=traj_mode)
        return depth_score, rgb_score, True

    @torch.no_grad()
    def denoise(self):
        print("start denoising free views..")
        view_graph = json.load(
            open(os.path.join(self.eval_dir, f"view_graph.json"), "r"))
        self.run_difix(view_graph)

    def vis(self):
        self.setup_visualizer()
        candidates = torch.load(os.path.join(
            self.eval_dir, 'cameras.pt'), weights_only=False)

        ref_camera = self.visualizer.cameras[1]
        H, W = ref_camera.height, ref_camera.width
        fy = ref_camera.params[1]
        FRUSTUM_COLOR = compute_rainbow_color(group_id=5)

        for idx, cam in enumerate(candidates):
            cam_to_world = cam.world_to_camera.detach().cpu().numpy()[0]
            vis_image = iio.imread(cam.image_path)
            self.visualizer.add_frame(
                len(self.visualizer.images) + idx,
                H, W, fy, fy,
                qvec_wxyz=cam_to_world.rotation().wxyz,
                position=cam_to_world.translation(),
                image=vis_image,
                color=FRUSTUM_COLOR,
            )
            time.sleep(0.1)

    @torch.no_grad()
    def run_difix(self, view_graph=None):
        freeviews_dir = os.path.join(self.eval_dir, "freeviews")
        if os.path.exists(f"{freeviews_dir}"):
            os.system(f"rm -rf {freeviews_dir}")
        os.makedirs(f"{freeviews_dir}/ref/", exist_ok=True)
        os.makedirs(f"{freeviews_dir}/difix/", exist_ok=True)
        if self.app_module is not None:
            for app_i in range(2):
                os.makedirs(f"{freeviews_dir}/difix_{app_i+1}/", exist_ok=True)

        orig_candidates = torch.load(os.path.join(
            self.eval_dir, 'cameras.pt'), weights_only=False)
        image_poses = [cam.cam_to_world.cpu().numpy()
                       for cam in orig_candidates]

        train_poses = torch.stack(
            [cam.cam_to_world for cam in self.train_cameras])
        train_poses = train_poses.cpu().numpy()
        ref_trainid = self.interpolator.find_nearest_assignments(
            train_poses, image_poses)
        candidates = []

        for ci, cam in enumerate(orig_candidates):
            i = int(cam.image_index.split("_")[1])
            candidate = copy.deepcopy(cam)
            image_path = os.path.join(self.eval_dir, cam.image_path)
            assert os.path.exists(image_path), f"{image_path}"
            image = Image.open(image_path).convert("RGB")

            ref_image = self.get_ref_fromVG(candidate.image_index, view_graph)
            if ref_image is None:
                ref_image = Image.open(
                    self.train_cameras[ref_trainid[ci]].image_path).convert("RGB")
            ref_image = ref_image.resize(image.size, Image.BILINEAR)
            # ref_image.save(f"{freeviews_dir}/ref/{i:03d}.png")

            output_image = self.difix(
                prompt="remove degradation",
                image=image,
                ref_image=ref_image,
                num_images_per_prompt=1,
                num_inference_steps=1,
                timesteps=[199],
                guidance_scale=0.0,
            ).images[0]
            output_image = output_image.resize(image.size, Image.LANCZOS)
            output_image.save(f"{freeviews_dir}/difix/{i:03d}.png")

            # update camera
            candidate.image_path = f"{freeviews_dir}/difix/{i:03d}.png"
            # candidate.image_index = f"fv_{i}"
            candidates.append(candidate)
            torch.save(candidates, f"{self.eval_dir}/cameras_difix.pt")

            # denoise for appearance variants
            if self.app_module is not None:
                # [-4, -2, 2, 4, 6]
                for app_i in range(2):
                    image_path = os.path.join(
                        self.eval_dir, cam.image_path.replace('rgb', f'rgb_{app_i + 1}'))
                    assert os.path.exists(image_path)
                    image = Image.open(image_path).convert("RGB")
                    output_image = self.difix(
                        prompt="remove degradation",
                        image=image,
                        ref_image=ref_image,
                        num_images_per_prompt=1,
                        num_inference_steps=1,
                        timesteps=[199],
                        guidance_scale=0.0,
                    ).images[0]
                    output_image = output_image.resize(
                        image.size, Image.LANCZOS)
                    output_image.save(
                        f"{freeviews_dir}/difix_{app_i + 1}/{i:03d}.png")

    @torch.no_grad()
    def get_ref_fromVG(self, ind, view_graph):
        neigbours = view_graph[ind]
        if len(neigbours) == 0:
            return None

        success = False
        num_try = 0
        while not success and num_try < 2:
            for n in neigbours:
                nid = n[0]
                if isinstance(nid, int) or nid.find("fv") < 0:
                    success = True
                    break
            if num_try >= len(neigbours):
                break
            neigbours = view_graph[str(neigbours[num_try][0])]
            num_try += 1

        if success:
            ref_path = self.train_cameras[int(nid)].image_path
        else:
            neigbours = view_graph[ind]
            neigbour_id = neigbours[0][0].split("_")[1]
            ref_path = f"{self.eval_dir}/rgb/{str(neigbour_id).zfill(3)}.png"
        ref_image = Image.open(ref_path).convert("RGB")

        return ref_image

    @torch.no_grad()
    def check_render_quality(
        self,
        colors,  # [H, W, 3]
        depth,
        black_threshold=0.2,
        depth_threshold=0.08,
    ):
        quality = 1
        mask = (colors[:, :, 0] == 0) & (
            colors[:, :, 1] == 0) & (colors[:, :, 2] == 0)
        black_ratio = mask.sum() / (mask.shape[0] * mask.shape[1])

        if black_ratio > black_threshold:
            quality = -1
            rgb_score = -1
        else:
            rgb_score = self.image_checker.score(colors)
            if rgb_score > 0:
                rgb_score = rgb_score / 100.

        # check based on depth map
        depth_score = filter_depth(
            depth, center_crop_ratio=self.config.sampler.depth_crop)
        if depth_score < depth_threshold:
            quality = -1
        else:
            quality = depth_score
        return quality, rgb_score


def filter_depth(depth, low_percentile=5, high_percentile=95, center_crop_ratio=0.6):
    H, W, _ = depth.shape
    if center_crop_ratio < 1.0:
        margin_h = int(H * (1 - center_crop_ratio) / 2)
        margin_w = int(W * (1 - center_crop_ratio) / 2)

        if margin_h < H // 2 and margin_w < W // 2:
            cropped_depth = depth[margin_h:H - margin_h, margin_w:W - margin_w]
        else:
            cropped_depth = depth
    else:
        cropped_depth = depth

    valid_depth = cropped_depth[cropped_depth > 0].flatten()
    valid_pixels_ratio = valid_depth.numel() / cropped_depth.numel()
    if valid_pixels_ratio < 0.1:
        return 0.

    min_val = torch.min(valid_depth)
    max_val = torch.max(valid_depth)

    if (max_val - min_val).item() < 1e-6:
        return 0.

    normalized_depth = (valid_depth - min_val) / (max_val - min_val)
    sorted_depth = torch.sort(normalized_depth).values
    n_elements = sorted_depth.numel()
    low_idx = torch.clamp(
        torch.tensor((low_percentile / 100.0) * (n_elements - 1)),
        min=0,
        max=n_elements - 1
    ).long()
    high_idx = torch.clamp(
        torch.tensor((high_percentile / 100.0) * (n_elements - 1)),
        min=0,
        max=n_elements - 1
    ).long()

    p_low = sorted_depth[low_idx]
    p_high = sorted_depth[high_idx]
    percentile_range = p_high - p_low

    return percentile_range.item()


def dynamic_filtering(
    confidences: np.ndarray,
    aligned_cameras: list,
    selected_masks: list,
    target_count: int = 100,
    initial_threshold: float = 0.6,
    max_threshold: float = 0.8,
    threshold_step: float = 0.05,
) -> tuple[list, np.ndarray, list, list]:
    if len(confidences) == 0:
        return [], np.array([]), [], []

    rgb_scores = confidences[:, 2]
    current_threshold = initial_threshold

    original_indices = np.arange(len(confidences))

    while True:
        current_selection_mask = (rgb_scores <= current_threshold)
        current_indices = original_indices[current_selection_mask]

        num_selected = len(current_indices)
        if num_selected >= target_count:
            final_indices = current_indices
            break

        if current_threshold >= max_threshold:
            final_indices = current_indices
            break

        current_threshold += threshold_step
        current_threshold = min(current_threshold, max_threshold)

    final_aligned_cameras = []
    final_node_names = []
    final_selected_masks = []
    final_confidences = []
    for i in final_indices:
        select_cam = aligned_cameras[i]
        cam_id = select_cam.image_index
        final_node_names.append(cam_id)
        final_aligned_cameras.append(select_cam)
        final_selected_masks.append(selected_masks[i])
        final_confidences.append([cam_id] + [str(c) for c in confidences[i]])

    return (
        final_aligned_cameras, final_confidences, final_node_names, final_selected_masks
    )
