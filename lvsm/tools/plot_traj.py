# modified from: https://github.com/hehao13/CameraCtrl/blob/main/tools/visualize_trajectory.py
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import json, os, sys
BASE_DIR = os.path.abspath(os.path.join( os.path.dirname( __file__ ), '..' ))
sys.path.append(BASE_DIR)

from data.dataset_scene import extract_pose_matrix

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

    def extrinsic2pyramid(self, extrinsic, color_value=0.5, hw_ratio=0.75, base_xval=0.1, zval=0.3):
        """
        Draw a small camera frustum/pyramid to represent the extrinsic matrix.
        extrinsic : (4,4) camera-to-world transform
        color_value : float in [0,1], mapped to a color via colormap
        hw_ratio : The aspect ratio of the camera plane
        base_xval, zval : size/length scalars for the frustum drawing
        """
        vertex_std = np.array([
            [0, 0, 0, 1],
            [ base_xval, -base_xval * hw_ratio, zval, 1],
            [ base_xval,  base_xval * hw_ratio, zval, 1],
            [-base_xval,  base_xval * hw_ratio, zval, 1],
            [-base_xval, -base_xval * hw_ratio, zval, 1]
        ])
        # Transform these points by the given extrinsic (camera-to-world).
        vertex_transformed = vertex_std @ extrinsic.T

        # Create triangular faces for the frustum
        meshes = [
            [vertex_transformed[0, :-1], vertex_transformed[1, :-1], vertex_transformed[2, :-1]],
            [vertex_transformed[0, :-1], vertex_transformed[2, :-1], vertex_transformed[3, :-1]],
            [vertex_transformed[0, :-1], vertex_transformed[3, :-1], vertex_transformed[4, :-1]],
            [vertex_transformed[0, :-1], vertex_transformed[4, :-1], vertex_transformed[1, :-1]],
            [vertex_transformed[1, :-1], vertex_transformed[2, :-1], vertex_transformed[3, :-1], vertex_transformed[4, :-1]]
        ]

        color = plt.cm.rainbow(color_value)
        self.ax.add_collection3d(
            Poly3DCollection(meshes, facecolors=color, linewidths=0.5, edgecolors=color, alpha=0.4)
        )

    def colorbar(self, max_value, interval=5):
        cmap = mpl.cm.rainbow
        norm = mpl.colors.Normalize(vmin=0, vmax=max_value*interval)
        # we need set the label correct as the cam traj len is not equal to the frame number
        # change frame number here I think

        cb = self.fig.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            ax=self.ax,
            orientation='vertical',
            label='Frame Number'
        )
        ticks = np.arange(0, max_value*interval + 1, interval)
        cb.set_ticks(ticks)
        cb.set_ticklabels([str(int(t)) for t in ticks])

    def show(self, save_path=None):
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, format='jpg', dpi=300)
        plt.show()

def load_w2c_4x4_entries(json_path):
    """
    Load w2c transforms from a JSON file where each frame has a 4x4 'w2c' array.
    Return a list of (4,4) transforms in camera-to-world format (i.e., invert each).
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    frames = data.get("frames", [])
    camera = data.get("camera", None)
    if len(frames) != 0:
        print("LVSM pose format.")
        c2w_matrices = []
        for idx, frame in enumerate(frames):
            w2c = frame.get("w2c", None)
            if w2c and len(w2c) == 4 and all(len(r) == 4 for r in w2c):
                w2c_mat = np.array(w2c, dtype=np.float32)
                # Convert from world->camera to camera->world
                c2w_mat = np.linalg.inv(w2c_mat)
                c2w_matrices.append(c2w_mat)
    elif camera is not None:        
        print("Our Kubric pose format.")
        poses = extract_pose_matrix(data)
        c2w_matrices = []
        
        for idx, w2c in enumerate(poses):
            w2c_mat = np.array(w2c, dtype=np.float32)
            # Convert from world->camera to camera->world
            c2w_mat = np.linalg.inv(w2c_mat)
            c2w_matrices.append(c2w_mat)
    return c2w_matrices

def filter_c2ws_by_interval_and_distance(c2ws, interval=1, dist_thresh=0.0):
    """
    1) Keep every 'interval'-th pose.
    2) Also skip poses if the distance from the last kept pose is below dist_thresh.
    """
    if not c2ws:
        return []

    filtered = [c2ws[0]]
    last_center = c2ws[0][:3, 3]

    for i in range(1, len(c2ws)):
        if i % interval != 0:
            continue
        center = c2ws[i][:3, 3]
        dist = np.linalg.norm(center - last_center)
        if dist >= dist_thresh:
            filtered.append(c2ws[i])
            last_center = center
    return filtered

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', type=str, default='/home/qingwen/workspace/LVSM/data/davis/metadata/hike.json')
    parser.add_argument('--x_min', type=float, default=-1.00)
    parser.add_argument('--x_max', type=float, default=1.00)
    parser.add_argument('--y_min', type=float, default=-1.00)
    parser.add_argument('--y_max', type=float, default=1.00)
    parser.add_argument('--z_min', type=float, default=-1.00)
    parser.add_argument('--z_max', type=float, default=1.00)
    parser.add_argument('--normalize_scale', action='store_true', help='Whether to scale positions to a unit range')
    parser.add_argument('--save_path', type=str, default='', help='Optional path for saving the figure')
    parser.add_argument('--interval', type=int, default=10, help='keep every nth frame')
    parser.add_argument('--dist_thresh', type=float, default=0.0, help='minimum distance to last plotted camera')
    args = parser.parse_args()

    # 1) Load camera-to-world transforms
    c2ws = load_w2c_4x4_entries(args.json_path)

    # 2) Filter frames by interval and distance threshold
    c2ws = filter_c2ws_by_interval_and_distance(c2ws, args.interval, args.dist_thresh)

    # 3) Optionally, normalize the translation for a more compact display
    if c2ws and args.normalize_scale:
        centers = [mat[:3, 3] for mat in c2ws]
        centers = np.array(centers)
        mean_center = np.mean(centers, axis=0)
        centers -= mean_center
        max_dist = np.max(np.linalg.norm(centers, axis=1))
        if max_dist > 1e-9:
            centers /= max_dist
        # Put back the adjusted translations
        for i, mat in enumerate(c2ws):
            c2ws[i][:3, 3] = centers[i]

    # 4) Create a visualizer and plot the resulting cameras
    viz = CameraPoseVisualizer([args.x_min, args.x_max], [args.y_min, args.y_max], [args.z_min, args.z_max])

    max_idx = max(len(c2ws) - 1, 1)
    for i, c2w in enumerate(c2ws):
        color_val = i / float(max_idx)
        viz.extrinsic2pyramid(c2w, color_val, hw_ratio=9.0 / 16.0, base_xval=0.1, zval=0.3)

    # 5) Add colorbar and show/save
    viz.colorbar(len(c2ws), interval=args.interval)
    viz.show(args.save_path)

if __name__ == '__main__':
    main()