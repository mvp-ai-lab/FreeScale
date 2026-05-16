import random
import time
from pathlib import Path
from typing import List, Optional, TypedDict

import imageio.v3 as iio
import numpy as np
import numpy.typing as npt
import tyro
from tqdm.auto import tqdm

import viser
import viser.transforms as vtf
from viser.extras.colmap import (
    read_cameras_binary,
    read_images_binary,
    read_points3d_binary,
)
from plyfile import PlyData
# from gs.datasets.normalize import (
#     transform_cameras,
#     transform_points,
# )


class SplatFile(TypedDict):
    """Data loaded from an antimatter15-style splat file."""
    centers: npt.NDArray[np.floating]
    rgbs: npt.NDArray[np.floating]
    opacities: npt.NDArray[np.floating]
    covariances: npt.NDArray[np.floating]


class GaussianSplattingVisualizer:
    def __init__(
        self,
        colmap_path: Path,
        images_path: Path,
        splats_path: Path,
        downsample_factor: int = 4,
        port: int = 1122,
        transform: Optional[vtf.SE3] = None,
        ply_path: Optional[Path] = None
    ):
        # Initialize server
        self.server = viser.ViserServer(port=port)

        # Store paths
        self.colmap_path = colmap_path
        self.images_path = images_path
        self.splats_path = splats_path
        self.downsample_factor = downsample_factor
        

        # Initialize state
        self.gs_handle: Optional[viser.GaussianSplatHandle] = None
        self.point_cloud: Optional[viser.PointCloudHandle] = None
        self.frames: List[viser.FrameHandle] = []
        self.frustums: List[viser.CameraFrustumHandle] = []
        self.need_update = False
        self.frustum_size = 0.15  # Default frustum size

        self.transform = transform

        # add ply load and GUI elements
        self.ply_path = ply_path
        self.ply_handle: Optional[viser.PointCloudHandle] = None
        self.gui_ply = None
        self.gui_ply_size = None

        # Initialize GUI elements
        self.gui_points = None
        self.gui_point_size = None
        self.gui_frames = None
        self.gui_reset_up = None
        self.gui_splat_scale = None
        self.gui_splat_opacity = None
        self.gui_show_splats = None
        self.gui_frustum_size = None  # Frustum size control
        self.status_text = self.server.gui.add_markdown("🟢 Ready")

        # Load data
        self.load_data()

        # Setup UI
        self.setup_ui()
   
    def setup_ui(self):
        dark_mode = self.server.gui.add_checkbox(
            "Dark mode", initial_value=True)
        # show_logo = self.server.gui.add_checkbox("Show logo", initial_value=True)
        show_share_button = self.server.gui.add_checkbox(
            "Show share button", initial_value=True)
        brand_color = self.server.gui.add_rgb("Brand color", (230, 180, 30))
        control_layout = self.server.gui.add_dropdown(
            "Control layout", ("floating", "fixed", "collapsible")
        )
        control_width = self.server.gui.add_dropdown(
            "Control width", ("small", "medium", "large"), initial_value="medium"
        )
        synchronize = self.server.gui.add_button(
            "Apply theme", icon=viser.Icon.CHECK)

        def synchronize_theme() -> None:
            self.server.gui.configure_theme(
                titlebar_content=None,  # titlebar_theme if titlebar.value else None,
                control_layout=control_layout.value,
                control_width=control_width.value,
                dark_mode=dark_mode.value,
                # show_logo=show_logo.value,
                show_share_button=show_share_button.value,
                brand_color=brand_color.value,
            )

        synchronize.on_click(lambda _: synchronize_theme())
        synchronize_theme()

        # # Status indicator
        # self.status_text = self.server.gui.add_markdown("🟢 Ready")

        """Setup the user interface with file selection controls."""
        # Add a new folder for path controls
        with self.server.gui.add_folder("📂 Data Selection"):
            # COLMAP folder selection
            self.gui_colmap_path = self.server.gui.add_text(
                "COLMAP Folder",
                initial_value=str(
                    self.colmap_path) if self.colmap_path else "",
                hint="Path to COLMAP reconstruction folder"
            )
            self.gui_colmap_browse = self.server.gui.add_button("Browse...")

            # Images folder selection
            self.gui_images_path = self.server.gui.add_text(
                "Images Folder",
                initial_value=str(
                    self.images_path) if self.images_path else "",
                hint="Path to directory containing images"
            )
            self.gui_images_browse = self.server.gui.add_button("Browse...")

            # Splats file selection
            self.gui_splats_path = self.server.gui.add_text(
                "Splats File",
                initial_value=str(
                    self.splats_path) if self.splats_path else "",
                hint="Path to Gaussian splat file (.splat)"
            )
            self.gui_splats_browse = self.server.gui.add_button("Browse...")

            # Ply file selection
            self.gui_ply_path = self.server.gui.add_text(
                "PLY File",
                initial_value=str(
                    self.ply_path) if self.ply_path else "",
                hint="Path to Ext PointCloud file (.ply)"
            )

            # Load button
            self.gui_load = self.server.gui.add_button(
                "Load Data", color="green")
            self.gui_load.on_click(lambda _: self._load_data_from_ui())

            # Add manual path input instructions
            self.server.gui.add_markdown("""
                **Instructions:**
                1. Enter paths manually above
                2. Click "Load Data" to update visualization
                """)

        """Setup the user interface with organized controls."""
        # Main control panel
        with self.server.gui.add_folder("📷 Camera Controls"):
            # Frustum size control at the top for visibility
            self.gui_frustum_size = self.server.gui.add_slider(
                "Frustum Size",
                min=0.05,
                max=0.5,
                step=0.01,
                initial_value=self.frustum_size,
                hint="Adjust camera frustum visualization size"
            )

            @self.gui_frustum_size.on_update
            def _(event: viser.GuiEvent):
                self.frustum_size = self.gui_frustum_size.value
                self.update_frustum_sizes()
                self.status_text.text = f"🟢 Frustum size set to {self.frustum_size:.2f}"

            self.gui_frames = self.server.gui.add_slider(
                "Max Frames",
                min=1,
                max=min(len(self.images), 500),
                step=1,
                initial_value=min(len(self.images), 20),
                hint="Number of camera frustums to display"
            )
            self.gui_frames.on_update(
                lambda _: setattr(self, 'need_update', True))

            self.gui_reset_up = self.server.gui.add_button(
                "🔄 Reset Up Direction",
                hint="Align up direction with average camera orientation"
            )

            @self.gui_reset_up.on_click
            def _(_):
                self.reorient_scene()

        # [Rest of the UI setup remains the same...]
        # Point cloud controls
        with self.server.gui.add_folder("☁️ Point Cloud"):
            max_points = len(self.points3d) if hasattr(
                self, 'points3d') else 1000
            initial_points = min(
                max_points, 50_000) if max_points > 0 else 1000

            self.gui_points = self.server.gui.add_slider(
                "Max Points",
                min=1,
                max=len(self.points3d),
                step=1,
                # min(len(self.points3d), 50_000),
                initial_value=initial_points,
                hint=f"Number of sparse points to display (max {max_points})"
            )
            self.gui_point_size = self.server.gui.add_slider(
                "Point Size",
                min=0.01,
                max=0.1,
                step=0.001,
                initial_value=0.02,
                hint="Size of each sparse point"
            )
            self.gui_points.on_update(self.update_point_cloud)
            self.gui_point_size.on_update(self.update_point_cloud)

        # Splatting controls
        with self.server.gui.add_folder("🎨 Gaussian Splats"):
            self.gui_show_splats = self.server.gui.add_checkbox(
                "Show Splats",
                initial_value=True,
                hint="Toggle splat visibility"
            )

            @self.gui_show_splats.on_update
            def _(event: viser.GuiEvent):
                if self.gs_handle:
                    self.gs_handle.visible = self.gui_show_splats.value
                    status = "visible" if self.gui_show_splats.value else "hidden"
                    self.status_text.text = f"{'🟢' if self.gui_show_splats.value else '🔴'} Splats {status}"

            self.gui_splat_scale = self.server.gui.add_slider(
                "Splat Scale",
                min=0.5,
                max=2.0,
                step=0.1,
                initial_value=1.0,
                hint="Scale factor for splat rendering"
            )

            @self.gui_splat_scale.on_update
            def _(_):
                if self.gs_handle:
                    # Need to update the entire splat data with new scale
                    self.update_splat_scale(self.gui_splat_scale.value)
                    self.status_text.text = f"🟢 Splat scale set to {self.gui_splat_scale.value:.2f}"

            self.gui_splat_opacity = self.server.gui.add_slider(
                "Opacity",
                min=0.0,
                max=1.0,
                step=0.05,
                initial_value=1.0,
                hint="Overall opacity of splats"
            )

            @self.gui_splat_opacity.on_update
            def _(_):
                if self.gs_handle:
                    # Need to update the entire splat data with new opacity
                    self.update_splat_opacity(self.gui_splat_opacity.value)
                    self.status_text.text = f"🟢 Splat opacity set to {self.gui_splat_opacity.value:.2f}"

        # Ext pointcloud controls
        with self.server.gui.add_folder("☁️ Ext Point Cloud"):
            max_points = len(self.ply_data["points"]) if hasattr(
                self, 'ply_data') else 1000
            initial_points = min(
                max_points, 50_000) if max_points > 0 else 1000

            self.gui_ply = self.server.gui.add_slider(
                "Max Points",
                min=1,
                max=len(self.ply_data["points"]),
                step=1,
                initial_value=initial_points,
                hint=f"Number of sparse points to display (max {max_points})"
            )
            self.gui_ply_size = self.server.gui.add_slider(
                "Ext Point Size",
                min=0.01,
                max=0.1,
                step=0.001,
                initial_value=0.02,
                hint="Size of each sparse point"
            )
            self.gui_ply.on_update(self.update_ext_point_cloud)
            self.gui_ply_size.on_update(self.update_ext_point_cloud)


        # Initialize visualizations after UI is set up
        self.initialize_visualizations()
        

    def _load_data_from_ui(self):
        """Load data based on UI path selections."""
        try:
            self.status_text.text = "🟠 Validating paths..."

            # Get paths from UI
            colmap_path = Path(self.gui_colmap_path.value.strip())
            images_path = Path(self.gui_images_path.value.strip())
            splats_path = Path(self.gui_splats_path.value.strip())
            ply_path = Path(self.gui_ply_path.value.strip())

            # Validate paths
            required_colmap_files = [
                "cameras.bin", "images.bin", "points3D.bin"]
            for f in required_colmap_files:
                if not (colmap_path / f).exists():
                    raise ValueError(f"Missing COLMAP file: {colmap_path/f}")

            if not images_path.exists():
                raise ValueError(f"Images path does not exist: {images_path}")

            if not splats_path.exists():
                raise ValueError(f"Splats path does not exist: {splats_path}")

            # Store paths
            self.colmap_path = colmap_path
            self.images_path = images_path
            self.splats_path = splats_path
            self.ply_path = ply_path

            # Clear existing visualizations
            self._clear_visualizations()

            # Load new data
            self.load_data()
            self.initialize_visualizations()
            self.reorient_scene()

            self.status_text.text = "🟢 Data loaded successfully!"

        except Exception as e:
            self.status_text.text = f"🔴 Error loading data: {str(e)}"
            print(f"Error loading data: {e}")

    def _clear_visualizations(self):
        """Clear all existing visualizations."""
        if self.gs_handle:
            self.gs_handle.remove()
            self.gs_handle = None

        if self.point_cloud:
            self.point_cloud.remove()
            self.point_cloud = None

        for frame in self.frames:
            frame.remove()
        self.frames = []

        for frustum in self.frustums:
            frustum.remove()
        self.frustums = []

        if self.ply_handle:
            self.ply_handle.remove()
            self.ply_handle = None

    # Add these new methods to the class:
    def update_splat_scale(self, scale: float):
        """Update the scale of all splats."""
        if not hasattr(self, 'splat_data'):
            return

        # Create new covariances with applied scale
        scaled_covariances = self.splat_data["covariances"] * (scale ** 2)

        with self.server.atomic():  # Batch updates for better performance
            self.gs_handle.covariances = scaled_covariances

    def update_splat_opacity(self, opacity: float):
        """Update the opacity of all splats."""
        if not hasattr(self, 'splat_data'):
            return

        # Create new opacities
        new_opacities = np.full_like(self.splat_data["opacities"], opacity)

        with self.server.atomic():  # Batch updates for better performance
            self.gs_handle.opacities = new_opacities

    def update_frustum_sizes(self):
        """Update all frustum sizes based on current size setting."""
        if not self.frustums:
            return

        with self.server.atomic():
            for frustum in self.frustums:
                frustum.scale = self.frustum_size

    def visualize_frames(self):
        """Visualize camera frames and frustums."""
        if self.gui_frames is None:
            return

        # Remove existing frames and frustums
        for frame in self.frames:
            frame.remove()
        self.frames.clear()

        for frustum in self.frustums:
            frustum.remove()
        self.frustums.clear()

        # Select frames to display
        img_ids = [im.id for im in self.images.values()]
        random.shuffle(img_ids)
        img_ids = sorted(img_ids[:self.gui_frames.value])

        for img_id in tqdm(img_ids, desc="Loading frames"):
            img = self.images[img_id]
            cam = self.cameras[img.camera_id]

            # Skip missing images
            image_filename = self.images_path / img.name
            if not image_filename.exists():
                continue

            # Create frame
            T_world_camera = vtf.SE3.from_rotation_and_translation(
                vtf.SO3(img.qvec), img.tvec
            ).inverse()
            
            # align with reconstructed gaussians
            if self.transform is not None:
                T_world_camera_matrix = T_world_camera.as_matrix()
                T_world_camera_matrix = transform_cameras(self.transform, T_world_camera_matrix[None, ...])
                T_world_camera = vtf.SE3.from_matrix(T_world_camera_matrix[0])
                
            frame = self.server.scene.add_frame(
                f"/frames/frame_{img_id}",
                wxyz=T_world_camera.rotation().wxyz,
                position=T_world_camera.translation(),
                show_axes=False,
                axes_length=0.1,
                axes_radius=0.005,
            )
            self.frames.append(frame)

            # Add frustum with image (using current frustum size)
            if cam.model == "PINHOLE":
                H, W = cam.height, cam.width
                fy = cam.params[1]
                image = iio.imread(image_filename)
                image = image[::self.downsample_factor,
                              ::self.downsample_factor]
                frustum = self.server.scene.add_camera_frustum(
                    f"/frames/frame_{img_id}/frustum",
                    fov=2 * np.arctan2(H / 2, fy),
                    aspect=W / H,
                    scale=self.frustum_size,  # Use current size
                    image=image,
                    color=(200, 0, 0),
                )
                self.frustums.append(frustum)

                @frustum.on_click
                def _(_, frame=frame):
                    for client in self.server.get_clients().values():
                        client.camera.wxyz = frame.wxyz
                        client.camera.position = frame.position

    # [Rest of the methods remain unchanged...]
    def load_data(self):
        """Load all data sources."""
        # Load COLMAP data
        self.status_text.text = "🟠 Loading COLMAP data..."
        self.cameras = read_cameras_binary(self.colmap_path / "cameras.bin")
        self.images = read_images_binary(self.colmap_path / "images.bin")
        self.points3d = read_points3d_binary(self.colmap_path / "points3D.bin")

        # Process points
        self.points = np.array(
            [self.points3d[p_id].xyz for p_id in self.points3d])
        if self.transform is not None:
            self.points = transform_points(self.transform, self.points)
        self.colors = np.array(
            [self.points3d[p_id].rgb for p_id in self.points3d])
        
        # Load splat data
        self.status_text.text = "🟠 Loading splat data..."
        self.splat_data = self.load_splat_file(self.splats_path, center=False)

        # Load splat or .ply data
        self.status_text.text = "🟠 Loading ply data..."
        if self.ply_path is not None:
            self.ply_data = self.load_ply_file(self.ply_path, center=False)

    def load_splat_file(self, splat_path: Path, center: bool = False) -> SplatFile:
        """Load an antimatter15-style splat file."""
        start_time = time.time()
        splat_buffer = splat_path.read_bytes()
        bytes_per_gaussian = 3*4 + 3*4 + 4 + 4  # position + scale + rgba + ijkl(rot)
        assert len(splat_buffer) % bytes_per_gaussian == 0
        num_gaussians = len(splat_buffer) // bytes_per_gaussian

        splat_uint8 = np.frombuffer(splat_buffer, dtype=np.uint8).reshape(
            (num_gaussians, bytes_per_gaussian)
        )
        scales = splat_uint8[:, 12:24].copy().view(np.float32)
        wxyzs = splat_uint8[:, 28:32] / 255.0 * 2.0 - 1.0
        Rs = vtf.SO3(wxyzs).as_matrix()
        covariances = np.einsum(
            "nij,njk,nlk->nil", Rs, np.eye(3)[None,
                                              :, :] * scales[:, None, :] ** 2, Rs
        )
        centers = splat_uint8[:, 0:12].copy().view(np.float32)
        if center:
            centers -= np.mean(centers, axis=0, keepdims=True)

        print(
            f"Loaded {num_gaussians} gaussians in {time.time() - start_time:.2f}s")
        return {
            "centers": centers,
            "rgbs": splat_uint8[:, 24:27] / 255.0,
            "opacities": splat_uint8[:, 27:28] / 255.0,
            "covariances": covariances,
        }

    # def load_ply_file(self, ply_path: Path, center: bool = False) -> Optional[SplatFile]:
    #     if not ply_path.exists() or ply_path.suffix.lower() != ".ply":
    #         return None
    #     ply_data = PlyData.read(ply_path)
    #     vertices = ply_data['vertex']
    #     points = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    #     colors = np.vstack([vertices['color_0'], vertices['color_1'], vertices['color_2']]).T
    #     colors = (colors * 255).astype

    #     return {"points": points, "colors":colors}
    
    def load_ply_file(self, ply_path: Path, center: bool = False) -> Optional[SplatFile]:
        if not ply_path.exists() or ply_path.suffix.lower() != ".ply":
            print(f"No found ext point cloud from {ply_path}")
            return None
        ply_data = PlyData.read(ply_path)
        vertices = ply_data['vertex']
        points = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
        colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T

        return {"points": points, "colors":colors}


    def initialize_visualizations(self):
        """Initialize all visualization elements."""
        # Add transform controls
        self.server.scene.add_transform_controls("/transform")

        # Create Gaussian splats
        self.gs_handle = self.server.scene.add_gaussian_splats(
            "/splats",
            centers=self.splat_data["centers"],
            rgbs=self.splat_data["rgbs"],
            opacities=self.splat_data["opacities"],
            covariances=self.splat_data["covariances"],
            visible=self.gui_show_splats.value if self.gui_show_splats else True
        )

        # Create initial point cloud
        self.update_point_cloud()

        self.update_ext_point_cloud()

        # Visualize initial frames
        self.visualize_frames()


    def update_point_cloud(self, _=None):
        """Update the point cloud visualization."""
        if self.gui_points is None or self.gui_point_size is None:
            return

        # point_mask = np.random.choice(self.points.shape[0], self.gui_points.value, replace=False)
        # Get the requested number of points, ensuring it doesn't exceed available points
        requested_points = min(self.gui_points.value, len(self.points))

        # Only sample if we're showing fewer than total points
        if requested_points < len(self.points):
            point_mask = np.random.choice(
                len(self.points),
                requested_points,
                replace=False
            )
            points_to_show = self.points[point_mask]
            colors_to_show = self.colors[point_mask]
        else:
            points_to_show = self.points
            colors_to_show = self.colors

        if self.point_cloud is None:
            self.point_cloud = self.server.scene.add_point_cloud(
                "/colmap/pcd",
                points=points_to_show,
                colors=colors_to_show,
                point_size=self.gui_point_size.value,
            )
        else:
            with self.server.atomic():
                self.point_cloud.points = points_to_show
                self.point_cloud.colors = colors_to_show
                self.point_cloud.point_size = self.gui_point_size.value

    def update_ext_point_cloud(self):
        """Update the point cloud visualization."""
        if self.gui_ply is None or self.gui_ply_size is None:
            return
        num_ext_pcl = len(self.ply_data["points"])
        requested_points = min(self.gui_ply.value, num_ext_pcl)

        if requested_points < num_ext_pcl:
            point_mask = np.random.choice(
                num_ext_pcl,
                requested_points,
                replace=False
            )
            points_to_show = self.ply_data["points"][point_mask]
            colors_to_show = self.ply_data["colors"][point_mask]
        else:
            points_to_show = self.ply_data["points"]
            colors_to_show = self.ply_data["colors"]

        if self.ply_handle is None:
            self.ply_handle = self.server.scene.add_point_cloud(
                "/external/pcd",
                points=points_to_show,
                colors=colors_to_show,
                point_size=self.gui_ply_size.value,
            )
        else:
            with self.server.atomic():
                self.ply_handle.points = points_to_show
                self.ply_handle.colors = colors_to_show
                self.ply_handle.point_size = self.gui_ply_size.value

    def run(self):
        """Main application loop."""
        while True:
            if self.need_update:
                self.need_update = False
                self.visualize_frames()
            time.sleep(1e-3)


def main(
    colmap_path: Path = Path(
        "[scene_path]/sparse/0"),
    images_path: Path = Path(
        "[scene_path]/images"),
    splats_path: Path = Path(
        "[gs_path]/web_splat.splat"),
    downsample_factor: int = 2,
    ply_path: Optional[Path] = None
) -> None:
    """Main entry point for the visualizer."""
    ext_ply_path = Path("[scene_path]/sparse/0/points3D_ext.ply")
    visualizer = GaussianSplattingVisualizer(
        colmap_path=colmap_path,
        images_path=images_path,
        splats_path=splats_path,
        downsample_factor=downsample_factor,
        ply_path=ext_ply_path
    )
    visualizer.run()


if __name__ == "__main__":
    tyro.cli(main)