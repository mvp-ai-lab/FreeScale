# pylint: disable=[E1101,C0103]

from typing import List, Literal

import trimesh
import torch
import numpy as np

from sklearn.cluster import SpectralClustering, KMeans

from conerf.datasets.utils import (
    compute_bounding_box2D,
    compute_bounding_box2D_trimesh,
    expand_bounding_box,
    points_in_bbox2D,
)


def get_partition_axis(aabb: np.ndarray) -> int:
    A, B = aabb[0, :], aabb[1, :]
    len_x_axis = B[0] - A[0]
    len_y_axis = B[1] - A[1]

    if len_x_axis >= len_y_axis:
        return 0

    return 1


def Grid2DBiPartite(
    points2d: np.ndarray,
    bbox_min_height: int = -1.0,
    bbox_max_height: int = 1.0,
    p0 = 0.02,
    p1 = 0.98,
    num_blocks: int = 1,
):
    grid_cells = []
    aabb = compute_bounding_box2D(
        torch.from_numpy(points2d), [1.0, 1.0], bbox_min_height, bbox_max_height, p0, p1
    )
    grid_cells.append(aabb)

    while len(grid_cells) < num_blocks:
        num_current_cells = len(grid_cells)
        print(f"num_current_cells: {num_current_cells}")
        while num_current_cells > 0:
            current_grid_cell = grid_cells.pop(0)
            axis_index = get_partition_axis(current_grid_cell)
            num_current_cells -= 1

            A, B = current_grid_cell[0, :], current_grid_cell[1, :]
            for i in range(0, 2):
                if axis_index == 0: # Splitting along x-axis.
                    x_divisions = np.linspace(A[0], B[0], 2 + 1)
                    grid_A = np.array([[x_divisions[i], A[1]]])
                    grid_B = np.array([[x_divisions[i + 1], B[1]]])
                else: # Splitting along y-axis.
                    y_divisions = np.linspace(A[1], B[1], 2 + 1)
                    grid_A = np.array([[A[0], y_divisions[i]]])
                    grid_B = np.array([[B[0], y_divisions[i + 1]]])

                cell_box = np.concatenate([grid_A, grid_B], axis=0)
                # Recompute a more compact bounding box for each cell.
                in_cell_point_indices = points_in_bbox2D(points2d, cell_box)
                in_cell_point_coors = points2d[in_cell_point_indices]
                compact_cell_box = compute_bounding_box2D(
                    torch.from_numpy(in_cell_point_coors), [1.0, 1.0],
                    bbox_min_height, bbox_max_height, 0, 1,
                )
                grid_cells.append(compact_cell_box.numpy())

    return grid_cells


def Grid2DXY(
    points2d: np.ndarray,
    bbox_min_height: int = -1.0,
    bbox_max_height: int = 1.0,
    p0 = 0.02,
    p1 = 0.98,
    mx: int = 1,
    my: int = 1,
    use_prior_center: bool = False,
    transform_world_to_obb: np.ndarray = None,
):
    if transform_world_to_obb is None:
        _, transform_world_to_obb = compute_bounding_box2D_trimesh(
            points2d, bbox_min_height, bbox_max_height, p0, p1
        )
    obb_points2d = trimesh.transform_points(points2d, transform_world_to_obb)

    aabb = compute_bounding_box2D(
        torch.from_numpy(obb_points2d), [1.0, 1.0], bbox_min_height, bbox_max_height, p0, p1)
    A, B = aabb[0, :], aabb[1, :]

    grid_cells = []

    if use_prior_center and mx * my == 4:
        center_min = np.array([0, 0, bbox_min_height])
        center_max = np.array([0, 0, bbox_max_height])
        grid_cells.append(np.stack([A, center_max], axis=0))              # cell1
        grid_cells.append(np.stack([[A[0], 0, bbox_min_height],
                                    [0, B[1], bbox_max_height]], axis=0)) # cell2
        grid_cells.append(np.stack([[0, A[1], bbox_min_height],
                                    [B[0], 0, bbox_max_height]], axis=0)) # cell3
        grid_cells.append(np.stack([center_min, B], axis=0))              # cell4

        return grid_cells, transform_world_to_obb

    # (1) Split scene along x-axis.
    x_divisions = np.linspace(A[0], B[0], mx + 1)
    x_grid_cells = []
    for i in range(0, mx):
        grid_A = np.array([[x_divisions[i], A[1]]])
        grid_B = np.array([[x_divisions[i + 1], B[1]]])
        cell_box = np.concatenate([grid_A, grid_B], axis=0)
        in_cell_point_indices = points_in_bbox2D(obb_points2d, cell_box)
        in_cell_point_coors = obb_points2d[in_cell_point_indices]
        # Recompute a more compact bounding box for each cell.
        compact_cell_box = compute_bounding_box2D(
            torch.from_numpy(in_cell_point_coors), [1.0, 1.0],
            bbox_min_height, bbox_max_height, 0, 1,
        )
        x_grid_cells.append(compact_cell_box)

    # (2) Split each grid cells along y-axis.
    grid_cells = []
    for x_grid_cell in x_grid_cells:
        x_grid_A, x_grid_B = x_grid_cell[0, :], x_grid_cell[1, :]
        y_divisions = np.linspace(x_grid_A[1], x_grid_B[1], my + 1)
        for j in range(0, my):
            grid_A = np.array([[x_grid_A[0], y_divisions[j]]])
            grid_B = np.array([[x_grid_B[0], y_divisions[j + 1]]])
            cell_box = np.concatenate([grid_A, grid_B], axis=0)
            grid_cells.append(np.concatenate(
                [cell_box, np.array([[bbox_min_height], [bbox_max_height]])
            ], axis=-1))

    return grid_cells, transform_world_to_obb


def Grid2DClustering(
    points: np.ndarray,
    scale_factor: List = [1.2, 1.2],
    bbox_min_height: int = -1.0,
    bbox_max_height: int = 1.0,
    p0 = 0.02,
    p1 = 0.98,
    num_blocks: int = 1, # pylint: disable=W0613
    mx: int = 1,
    my: int = 1,
    use_prior_center: np.ndarray = None,
    transform_world_to_obb: np.ndarray = None,
) -> np.array:
    # points should be axis aligned such that its z-axis is aligned to real-world's gravity.
    points2d = points[:, :2]

    # if num_blocks > 1:
        # transform_obb_to_world = None
        # grid_cells = Grid2DBiPartite(
        # points2d, bbox_min_height, bbox_max_height, p0, p1, num_blocks)
    # else:
    grid_cells, transform_obb_to_world = Grid2DXY(
        points2d, bbox_min_height, bbox_max_height,
        p0, p1, mx, my, use_prior_center, transform_world_to_obb,
    )

    # Group points into each grid cell.
    num_block_points = [None] * len(grid_cells)
    labels = np.zeros(points.shape[0], dtype=np.uint8)
    for k, cell in enumerate(grid_cells):
        in_cell_point_indices = points_in_bbox2D(points2d, cell, transform_obb_to_world)
        labels[in_cell_point_indices] = k
        num_block_points[k] = in_cell_point_indices.shape[0]

    exp_grid_cells = []
    # Expand the original bounding box.
    avg_block_points = float(sum(num_block_points)) / len(num_block_points)
    for k, cell_box in enumerate(grid_cells):
        ratio_points = float(avg_block_points) / float(num_block_points[k])
        exp_scale_factor = scale_factor
        # # Smaller blocks should be expanded more aggressively.
        # if ratio_points >= 1.2:
        #     exp_scale_factor = [1.2 * factor for factor in scale_factor]
        print(f'aggressive expanding bounding box with scale factor: {exp_scale_factor}')

        exp_cell_box = expand_bounding_box(
            torch.from_numpy(cell_box[:, :2].reshape(-1)), scale_factor=exp_scale_factor)
        exp_cell_box = torch.concat(
            [exp_cell_box, torch.tensor([[bbox_min_height], [bbox_max_height]])
        ], dim=-1)
        exp_grid_cells.append(exp_cell_box.numpy())

    return labels, grid_cells, exp_grid_cells, transform_obb_to_world


def clustering(
    points: np.ndarray,
    num_clusters: int = 2,
    method: str = Literal["KMeans", "Spectral", "Grid2d"],
    **args
) -> np.array:
    """
    Args:
        params points: point positions in world frame, [N, 3]
        params num_cluster: number of clusters to partition
        params method: use 'KMeans' or 'Spectral'
    Return:
        cluster labels for corresponding camera poses.
    """

    if method == 'KMeans':
        clustering = KMeans(
            n_clusters=num_clusters,
            random_state=0,
            n_init="auto"
        ).fit(points)
        return clustering.labels_, None, None

    if method == 'Spectral':
        clustering = SpectralClustering(
            n_clusters=num_clusters,
            assign_labels='discretize',
            random_state=0
        ).fit(points)
        return clustering.labels_, None, None

    if method == 'Grid2d':
        return Grid2DClustering(points, num_blocks=num_clusters, **args)

    raise NotImplementedError
