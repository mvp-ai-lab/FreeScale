import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import json, os, sys
from pathlib import Path

import torch

from tools.camera_utils import _load_intrinsics_for_frame, generate_camera_trajectory

class CameraPoseVisualizer:
    def __init__(self, xlim, ylim, zlim):
        self.fig = plt.figure(figsize=(18, 7))
        self.ax = self.fig.add_subplot(projection='3d')
        self.plotly_data = None
        self.ax.set_aspect("auto")
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.set_zlim(zlim)
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')
        print('initialize camera pose visualizer')
        # Connect keyboard event
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        print("Press 'w' to save view, 'e' to load view")

    def extrinsic2pyramid(self, extrinsic, color_map='red', hw_ratio=9/16, base_xval=1, zval=3):
        vertex_std = np.array([[0, 0, 0, 1],
                               [base_xval, -base_xval * hw_ratio, zval, 1],
                               [base_xval, base_xval * hw_ratio, zval, 1],
                               [-base_xval, base_xval * hw_ratio, zval, 1],
                               [-base_xval, -base_xval * hw_ratio, zval, 1]])
        vertex_transformed = vertex_std @ extrinsic.T
        meshes = [[vertex_transformed[0, :-1], vertex_transformed[1][:-1], vertex_transformed[2, :-1]],
                            [vertex_transformed[0, :-1], vertex_transformed[2, :-1], vertex_transformed[3, :-1]],
                            [vertex_transformed[0, :-1], vertex_transformed[3, :-1], vertex_transformed[4, :-1]],
                            [vertex_transformed[0, :-1], vertex_transformed[4, :-1], vertex_transformed[1, :-1]],
                            [vertex_transformed[1, :-1], vertex_transformed[2, :-1], vertex_transformed[3, :-1], vertex_transformed[4, :-1]]]

        color = color_map if isinstance(color_map, str) else plt.cm.rainbow(color_map)

        self.ax.add_collection3d(
            Poly3DCollection(meshes, facecolors=color, linewidths=0.3, edgecolors=color, alpha=0.35))

    def customize_legend(self, list_label):
        list_handle = []
        for idx, label in enumerate(list_label):
            color = plt.cm.viridis(idx / len(list_label))
            patch = Patch(color=color, label=label)
            list_handle.append(patch)
        plt.legend(loc='right', bbox_to_anchor=(1.8, 0.5), handles=list_handle)

    def colorbar(self, max_frame_length):
        cmap = mpl.cm.rainbow
        norm = mpl.colors.Normalize(vmin=0, vmax=max_frame_length)
        self.fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=self.ax, orientation='vertical', label='Frame Number')

    def on_key_press(self, event):
        # Handle keyboard events
        if event.key == 'w':
            self.save_view()
        elif event.key == 'e':
            self.load_view()
            self.fig.canvas.draw()  # Refresh display

    def save_view(self, filename='assets/view_config.json'):
        view_params = {
            'elev': float(self.ax.elev),
            'azim': float(self.ax.azim),
            'roll': float(self.ax.roll),
            'xlim': [float(x) for x in self.ax.get_xlim()],
            'ylim': [float(y) for y in self.ax.get_ylim()],
            'zlim': [float(z) for z in self.ax.get_zlim()]
        }
        with open(filename, 'w') as f:
            json.dump(view_params, f, indent=2)
        print(f'View saved to {filename}')
        
    def load_view(self, filename='assets/view_config.json'):
        try:
            with open(filename, 'r') as f:
                view_params = json.load(f)
            self.ax.view_init(
                elev=view_params['elev'],
                azim=view_params['azim'],
                roll=view_params['roll']
            )
            self.ax.set_xlim(view_params['xlim'])
            self.ax.set_ylim(view_params['ylim'])
            self.ax.set_zlim(view_params['zlim'])
            print(f'View loaded from {filename}')
        except FileNotFoundError:
            print(f'No saved view found at {filename}')

    def show(self):
        plt.title('Extrinsic Parameters')
        # plt.savefig('extrinsic_parameters.jpg', format='jpg', dpi=300)
        plt.show()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', default='/home/ssd/qingwen/data/dynposepp/subset/pose', type=str, help='the path of pose files')
    parser.add_argument('--stride', type=int, default=4)
    parser.add_argument('--hw_ratio', default=9/16, type=float, help='the height over width of the film plane')
    parser.add_argument('--base_xval', type=float, default=0.08)
    parser.add_argument('--zval', type=float, default=0.15)    
    parser.add_argument('--x_min', type=float, default=-2)
    parser.add_argument('--x_max', type=float, default=2)
    parser.add_argument('--y_min', type=float, default=-2)
    parser.add_argument('--y_max', type=float, default=2)
    parser.add_argument('--z_min', type=float, default=-1.)
    parser.add_argument('--z_max', type=float, default=1)
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    pose_files = sorted(os.listdir(args.data_path))

    vis_pose = Path(args.data_path) / pose_files[0]
    intri = Path(args.data_path).parent / 'intrinsics' / pose_files[0]

    scene_data_dict = np.load(vis_pose)
    pose = scene_data_dict['data'] # N, 4, 4
    intri = _load_intrinsics_for_frame(intri, 0) # 3, 3
    inds = scene_data_dict['inds'] # N

    # maybe only for matplotlib visualization??
    # flip = np.eye(4)
    # flip[0, 0] = -1  # Flip X axis (or use index 1 for Y, 2 for Z)
    # pose = pose @ flip  # Shape: (N, 4, 4)


    vis_cameras = [pose[i] for i in range(0, len(inds), args.stride)]
    vis_intri = [intri for i in range(0, len(inds), args.stride)]
    vis_cameras = np.stack(vis_cameras, axis=0)
    vis_intri = np.stack(vis_intri, axis=0)
    print(f'visualizing {len(vis_cameras)} cameras from {vis_pose}, {vis_cameras.shape}')

    visualizer = CameraPoseVisualizer([args.x_min, args.x_max], [args.y_min, args.y_max], [args.z_min, args.z_max])
    for ind in range(len(vis_cameras)):
        c2w = vis_cameras[ind]
        visualizer.extrinsic2pyramid(c2w, ind / len(vis_cameras), hw_ratio=args.hw_ratio, base_xval=args.base_xval,
                                     zval=(args.zval))
    visualizer.colorbar(len(vis_cameras))

    # add target view here to have a check!
    # middle_pose = pose[10] # HRADCODE
    initial_cam_w2c_for_traj = [np.linalg.inv(vis_cameras[ind]) for ind in range(len(vis_cameras))]
    initial_cam_w2c_for_traj = initial_cam_w2c_for_traj[10]
    initial_cam_intrinsics_for_traj = vis_intri[10, ...]
    move_dis_between_two_frames = np.linalg.norm(vis_cameras[10, :3, 3] - vis_cameras[9, :3, 3])
    # print(f'move_dis_between_two_frames: {move_dis_between_two_frames}, pose0 : {vis_cameras[10, ...]}, pose1 : {vis_cameras[9, ...]}')
    generated_w2cs, generated_intrinsics = generate_camera_trajectory(
        trajectory_type="clockwise",
        initial_w2c=torch.from_numpy(initial_cam_w2c_for_traj).float(),
        initial_intrinsics=torch.from_numpy(initial_cam_intrinsics_for_traj).float(),
        num_frames=3,
        movement_distance=move_dis_between_two_frames,
        camera_rotation='center_facing',
        center_depth=1.0,
        device="cpu",
    )
    generated_w2cs = generated_w2cs.squeeze(0)[1:, ...] # N-1, 4, 4
    new_poses = [np.linalg.inv(generated_w2cs[i].cpu().numpy()) for i in range(generated_w2cs.shape[0])]
    print(f'adding {len(new_poses)} new views around the {10}-th view for checking')
    for ind in range(0, len(new_poses)):
        c2w = new_poses[ind]
        visualizer.extrinsic2pyramid(c2w, color_map='gray', hw_ratio=args.hw_ratio, base_xval=args.base_xval,
                                     zval=(args.zval))
    visualizer.show()