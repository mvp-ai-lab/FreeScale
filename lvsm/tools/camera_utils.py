# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import math
import numpy as np

def _intrinsics_from_fxfycxcy(fxfycxcy: np.ndarray) -> np.ndarray:
    fx, fy, cx, cy = [float(x) for x in fxfycxcy]
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    return K

def _load_intrinsics_for_frame(intrinsics_npz_path: str, frame_idx: int) -> np.ndarray:
    data = np.load(intrinsics_npz_path)
    inds = data["inds"]
    arr = data["data"]
    pos = int(np.searchsorted(inds, frame_idx))
    if not (0 <= pos < len(inds)) or int(inds[pos]) != int(frame_idx):
        raise FileNotFoundError(
            f"Intrinsics for frame {frame_idx} not found in {intrinsics_npz_path}"
        )
    item = arr[pos]
    if item.shape == (3, 3):
        K = item.astype(np.float32)
    elif item.shape[-1] == 4:
        K = _intrinsics_from_fxfycxcy(item)
    else:
        raise ValueError(
            f"Unsupported intrinsics format {item.shape} in {intrinsics_npz_path}"
        )
    return K

def apply_transformation(Bx4x4, another_matrix):
    B = Bx4x4.shape[0]
    if another_matrix.dim() == 2:
        another_matrix = another_matrix.unsqueeze(0).expand(B, -1, -1)  # Make another_matrix compatible with batch size
    transformed_matrix = torch.bmm(Bx4x4, another_matrix)  # Shape: (B, 4, 4)

    return transformed_matrix

def look_at_matrix(camera_pos, target, invert_pos=True):
    """Creates a 4x4 look-at matrix, keeping the camera pointing towards a target."""
    forward = (target - camera_pos).float()
    forward = forward / torch.norm(forward)

    up = torch.tensor([0.0, 1.0, 0.0], device=camera_pos.device)  # assuming Y-up coordinate system
    right = torch.cross(up, forward)
    right = right / torch.norm(right)
    up = torch.cross(forward, right)

    look_at = torch.eye(4, device=camera_pos.device)
    look_at[0, :3] = right
    look_at[1, :3] = up
    look_at[2, :3] = forward
    look_at[:3, 3] = (-camera_pos) if invert_pos else camera_pos

    return look_at

def create_horizontal_trajectory(
    world_to_camera_matrix, center_depth, positive=True, n_steps=13, distance=0.1, device="cuda", axis="x", camera_rotation="center_facing"
):
    look_at = torch.tensor([0.0, 0.0, center_depth]).to(device)
    # Spiral motion key points
    trajectory = []
    translation_positions = []
    initial_camera_pos = torch.tensor([0, 0, 0], device=device)

    for i in range(n_steps):
        if axis == "x": # pos - right
            x = i * distance * center_depth / n_steps * (1 if positive else -1)
            y = 0
            z = 0
        elif axis == "y": # pos - down
            x = 0
            y = i * distance * center_depth / n_steps * (1 if positive else -1)
            z = 0
        elif axis == "z": # pos - in
            x = 0
            y = 0
            z = i * distance * center_depth / n_steps * (1 if positive else -1)
        else:
            raise ValueError("Axis should be x, y or z")

        translation_positions.append(torch.tensor([x, y, z], device=device))

    for pos in translation_positions:
        camera_pos = initial_camera_pos + pos
        if camera_rotation == "trajectory_aligned":
            _look_at = look_at + pos * 2
        elif camera_rotation == "center_facing":
            _look_at = look_at
        elif camera_rotation == "no_rotation":
            _look_at = look_at + pos
        else:
            raise ValueError("Camera rotation should be center_facing or trajectory_aligned")
        view_matrix = look_at_matrix(camera_pos, _look_at)
        trajectory.append(view_matrix)
    trajectory = torch.stack(trajectory)
    return apply_transformation(trajectory, world_to_camera_matrix)


def create_spiral_trajectory(
    world_to_camera_matrix,
    center_depth,
    radius_x=0.03,
    radius_y=0.02,
    radius_z=0.0,
    positive=True,
    camera_rotation="center_facing",
    n_steps=13,
    device="cuda",
    start_from_zero=True,
    num_circles=1,
):

    look_at = torch.tensor([0.0, 0.0, center_depth]).to(device)

    # Spiral motion key points
    trajectory = []
    spiral_positions = []
    initial_camera_pos = torch.tensor([0, 0, 0], device=device)  # world_to_camera_matrix[:3, 3].clone()

    example_scale = 1.0

    theta_max = 2 * math.pi * num_circles

    for i in range(n_steps):
        # theta = 2 * math.pi * i / (n_steps-1)  # angle for each point
        theta = theta_max * i / (n_steps - 1)  # angle for each point
        if start_from_zero:
            x = radius_x * (math.cos(theta) - 1) * (1 if positive else -1) * (center_depth / example_scale)
        else:
            x = radius_x * (math.cos(theta)) * (center_depth / example_scale)

        y = radius_y * math.sin(theta) * (center_depth / example_scale)
        z = radius_z * math.sin(theta) * (center_depth / example_scale)
        spiral_positions.append(torch.tensor([x, y, z], device=device))

    for pos in spiral_positions:
        if camera_rotation == "center_facing":
            view_matrix = look_at_matrix(initial_camera_pos + pos, look_at)
        elif camera_rotation == "trajectory_aligned":
            view_matrix = look_at_matrix(initial_camera_pos + pos, look_at + pos * 2)
        elif camera_rotation == "no_rotation":
            view_matrix = look_at_matrix(initial_camera_pos + pos, look_at + pos)
        else:
            raise ValueError("Camera rotation should be center_facing, trajectory_aligned or no_rotation")
        trajectory.append(view_matrix)
    trajectory = torch.stack(trajectory)
    return apply_transformation(trajectory, world_to_camera_matrix)


def generate_camera_trajectory(
    trajectory_type: str,
    initial_w2c: torch.Tensor,  # Shape: (4, 4)
    initial_intrinsics: torch.Tensor,  # Shape: (3, 3)
    num_frames: int,
    movement_distance: float,
    camera_rotation: str,
    center_depth: float = 1.0,
    device: str = "cuda",
):
    """
    Generates a sequence of camera poses (world-to-camera matrices) and intrinsics
    for a specified trajectory type.

    Args:
        trajectory_type: Type of trajectory (e.g., "left", "right", "up", "down", "zoom_in", "zoom_out").
        initial_w2c: Initial world-to-camera matrix (4x4 tensor or num_framesx4x4 tensor).
        initial_intrinsics: Camera intrinsics matrix (3x3 tensor or num_framesx3x3 tensor).
        num_frames: Number of frames (steps) in the trajectory.
        movement_distance: Distance factor for the camera movement.
        camera_rotation: Type of camera rotation ('center_facing', 'no_rotation', 'trajectory_aligned').
        center_depth: Depth of the center point the camera might focus on.
        device: Computation device ("cuda" or "cpu").

    Returns:
        A tuple (generated_w2cs, generated_intrinsics):
        - generated_w2cs: Batch of world-to-camera matrices for the trajectory (1, num_frames, 4, 4 tensor).
        - generated_intrinsics: Batch of camera intrinsics for the trajectory (1, num_frames, 3, 3 tensor).
    """
    if trajectory_type in ["clockwise", "counterclockwise"]:
        new_w2cs_seq = create_spiral_trajectory(
            world_to_camera_matrix=initial_w2c,
            center_depth=center_depth,
            n_steps=num_frames,
            positive=trajectory_type == "clockwise",
            device=device,
            camera_rotation=camera_rotation,
            radius_x=movement_distance,
            radius_y=movement_distance,
        )
    else:
        if trajectory_type == "left":
            positive = False
            axis = "x"
        elif trajectory_type == "right":
            positive = True
            axis = "x"
        elif trajectory_type == "up":
            positive = False  # Assuming 'up' means camera moves in negative y direction if y points down
            axis = "y"
        elif trajectory_type == "down":
            positive = True # Assuming 'down' means camera moves in positive y direction if y points down
            axis = "y"
        elif trajectory_type == "zoom_in":
            positive = True  # Assuming 'zoom_in' means camera moves in positive z direction (forward)
            axis = "z"
        elif trajectory_type == "zoom_out":
            positive = False # Assuming 'zoom_out' means camera moves in negative z direction (backward)
            axis = "z"
        else:
            raise ValueError(f"Unsupported trajectory type: {trajectory_type}")

        # Generate world-to-camera matrices using create_horizontal_trajectory
        new_w2cs_seq = create_horizontal_trajectory(
            world_to_camera_matrix=initial_w2c,
            center_depth=center_depth,
            n_steps=num_frames,
            positive=positive,
            axis=axis,
            distance=movement_distance,
            device=device,
            camera_rotation=camera_rotation,
        )

    generated_w2cs = new_w2cs_seq.unsqueeze(0)  # Shape: [1, num_frames, 4, 4]
    if initial_intrinsics.dim() == 2:
        generated_intrinsics = initial_intrinsics.unsqueeze(0).unsqueeze(0).repeat(1, num_frames, 1, 1)
    else:
        generated_intrinsics = initial_intrinsics.unsqueeze(0)

    return generated_w2cs, generated_intrinsics
