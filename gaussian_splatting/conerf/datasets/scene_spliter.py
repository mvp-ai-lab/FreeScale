# pylint: disable=[C0103,E0401,R0903]

import os
from typing import Dict

import numpy as np

from conerf.geometry.cluster import clustering
from conerf.pycolmap.pycolmap.scene_manager import SceneManager


class SceneSplitter: # pylint: disable=R0903
    """
    Split a scene reconstructed by COLMAP according to its camera poses or
    sparse 3D points.
    """

    def __init__(
        self,
        scene_manager: SceneManager,
    ) -> None:
        self.images = scene_manager.images
        # self.points3D = scene_manager.points3D
        self.point3D_id_to_images = scene_manager.point3D_id_to_images
        # self.point3D_id_to_point3D_idx = scene_manager.point3D_id_to_point3D_idx
        self.point3D_idx_to_point3D_id = scene_manager.point3D_idx_to_point3D_id

    def _save_split_result(self, labels: Dict, save_dir: str) -> None: # pylint: disable=R0201
        split_result_file = open(
            os.path.join(save_dir, "cluster.txt"),
            "w",
            encoding="utf-8",
        )

        for image_id, label in enumerate(labels):
            print(f'{image_id} {label}', file=split_result_file)

        split_result_file.close()

    def split(
        self,
        camtoworlds: np.ndarray = None,
        points3D: np.ndarray = None,
        split_type: str = 'camera',
        num_blocks: int = 1,
        method: str = "KMeans",
        save_dir: str = "",
    ) -> Dict:
        """Main function to split the scene."""
        if split_type == 'camera':
            labels, _, _ = clustering(
                camtoworlds[..., :3, -1], num_clusters=num_blocks, method=method)
        elif split_type == 'point':
            point_labels, _, _ = clustering(
                points3D, num_clusters=num_blocks, method=method)
            labels = dict()

            # Grouping images according to their point->images association.
            for point_idx, point_label in enumerate(point_labels):
                point_id = self.point3D_idx_to_point3D_id[point_idx]
                image_ids = self.point3D_id_to_images[point_id][..., 0]
                for image_id in image_ids:
                    labels[image_id - 1] = point_label
        else:
            raise NotImplementedError

        self._save_split_result(labels=labels, save_dir=save_dir)

        return labels
