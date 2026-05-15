# pylint: disable=[E1101,C0206,E0401,C0103]

from conerf.datasets.dataset_base import DatasetBase
from conerf.datasets.load_colmap import load_colmap


class SubjectLoader(DatasetBase):
    OPENGL_CAMERA = False
    DATA_TYPE = "REAL_WORLD"

    def __init__(
        self,
        subject_id: str,
        root_fp: str,
        split: str,
        data_split_json: str = "",
        color_bkgd_aug: str = "white",
        num_rays: int = None,
        near: float = None,
        far: float = None,
        batch_over_images: bool = True,
        factor: int = 1,
        val_interval: int = 0,
        multi_blocks: bool = False,
        num_blocks: int = 1,
        **kwargs
    ) -> None:
        super().__init__(
            subject_id,
            root_fp,
            split,
            data_split_json,
            color_bkgd_aug,
            num_rays,
            near,
            far,
            batch_over_images,
            factor,
            val_interval,
            multi_blocks,
            num_blocks,
            **kwargs
        )

    def load_data(
        self,
        root_fp: str,
        subject_id: str,
        split: str,
        factor: int = 1,
        num_blocks: int = 1,
    ):
        return load_colmap(
            root_fp, subject_id, split, factor,
            self.val_interval, self.multi_blocks, num_blocks,
            self.bbox_scale_factor, self.scale, self.rotate, self.use_manhattan_world,
            self.model_folder, self.load_specified_images,
            self.load_normal, self.load_depth, self.load_mask,
            self.mx, self.my,
        )
