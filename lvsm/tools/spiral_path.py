import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

def apply_transformation(Bx4x4, another_matrix):
    B = Bx4x4.shape[0]
    if another_matrix.dim() == 2:
        another_matrix = another_matrix.unsqueeze(0).expand(B, -1, -1)  # Make another_matrix compatible with batch size
    transformed_matrix = torch.bmm(Bx4x4, another_matrix)  # Shape: (B, 4, 4)
    return transformed_matrix

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
    if not torch.cuda.is_available():
        device = "cpu"
    
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

def test_spiral_trajectory():
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 创建一个简单的world_to_camera_matrix (单位矩阵)
    world_to_camera_matrix = torch.eye(4, device=device)
    
    # 测试不同的参数组合
    test_configs = [
        {
            "name": "Default Small Spiral",
            "center_depth": 5.0,
            "radius_x": 0.5,
            "radius_y": 0.3,
            "radius_z": 0.0,
            "n_steps": 20,
            "camera_rotation": "center_facing",
            "start_from_zero": True,
            "num_circles": 1
        },
        {
            "name": "Two Circles with Z motion",
            "center_depth": 3.0,
            "radius_x": 0.8,
            "radius_y": 0.6,
            "radius_z": 0.2,
            "n_steps": 30,
            "camera_rotation": "center_facing",
            "start_from_zero": False,
            "num_circles": 2
        },
        {
            "name": "Trajectory Aligned",
            "center_depth": 4.0,
            "radius_x": 0.6,
            "radius_y": 0.4,
            "radius_z": 0.1,
            "n_steps": 25,
            "camera_rotation": "trajectory_aligned",
            "start_from_zero": True,
            "num_circles": 1.5
        }
    ]
    
    fig = plt.figure(figsize=(20, 12))
    
    for idx, config in enumerate(test_configs):
        print(f"\nTesting: {config['name']}")
        
        # 生成轨迹
        trajectory = create_spiral_trajectory(
            world_to_camera_matrix,
            config["center_depth"],
            radius_x=config["radius_x"],
            radius_y=config["radius_y"], 
            radius_z=config["radius_z"],
            camera_rotation=config["camera_rotation"],
            n_steps=config["n_steps"],
            device=device,
            start_from_zero=config["start_from_zero"],
            num_circles=config["num_circles"]
        )
        
        # 转换为numpy进行可视化
        trajectory_np = trajectory.cpu().numpy()
        
        # 提取相机位置（注意这里可能需要根据你的look_at_matrix实现调整）
        # 如果invert_pos=True，位置存储为负值
        camera_positions = -trajectory_np[:, :3, 3]  # 假设invert_pos=True
        
        # 提取目标点
        look_at_point = np.array([0.0, 0.0, config["center_depth"]])
        
        # 3D轨迹可视化
        ax = fig.add_subplot(2, 3, idx*2 + 1, projection='3d')
        
        # 绘制轨迹
        ax.plot(camera_positions[:, 0], camera_positions[:, 1], camera_positions[:, 2], 
                'b-', linewidth=2, label='Camera Path')
        ax.scatter(*camera_positions[0], color='green', s=100, label='Start')
        ax.scatter(*camera_positions[-1], color='red', s=100, label='End')
        ax.scatter(*look_at_point, color='orange', s=200, marker='*', label='Look-at Target')
        
        # 绘制一些相机朝向箭头
        step = max(1, len(camera_positions) // 8)
        for i in range(0, len(camera_positions), step):
            cam_pos = camera_positions[i]
            
            # 计算朝向向量
            if config["camera_rotation"] == "center_facing":
                look_dir = look_at_point - cam_pos
            elif config["camera_rotation"] == "trajectory_aligned":
                spiral_pos = cam_pos  # 相对于初始位置的偏移
                look_dir = (look_at_point + spiral_pos * 2) - cam_pos
            else:  # no_rotation
                spiral_pos = cam_pos
                look_dir = (look_at_point + spiral_pos) - cam_pos
                
            look_dir = look_dir / np.linalg.norm(look_dir) * 0.5
            
            ax.quiver(cam_pos[0], cam_pos[1], cam_pos[2],
                     look_dir[0], look_dir[1], look_dir[2],
                     color='gray', alpha=0.6, arrow_length_ratio=0.1)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'{config["name"]}\n3D Trajectory')
        ax.legend()
        ax.grid(True)
        
        # XY平面投影
        ax2 = fig.add_subplot(2, 3, idx*2 + 2)
        ax2.plot(camera_positions[:, 0], camera_positions[:, 1], 'b-', linewidth=2)
        ax2.scatter(*camera_positions[0][:2], color='green', s=100, label='Start')
        ax2.scatter(*camera_positions[-1][:2], color='red', s=100, label='End')
        ax2.scatter(*look_at_point[:2], color='orange', s=200, marker='*', label='Target')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_title(f'{config["name"]}\nTop View (XY)')
        ax2.legend()
        ax2.grid(True)
        ax2.axis('equal')
        
        # 打印统计信息
        distances = np.linalg.norm(camera_positions - look_at_point, axis=1)
        print(f"  Camera positions shape: {camera_positions.shape}")
        print(f"  Distance to target: min={distances.min():.3f}, max={distances.max():.3f}, mean={distances.mean():.3f}")
        print(f"  X range: [{camera_positions[:, 0].min():.3f}, {camera_positions[:, 0].max():.3f}]")
        print(f"  Y range: [{camera_positions[:, 1].min():.3f}, {camera_positions[:, 1].max():.3f}]")
        print(f"  Z range: [{camera_positions[:, 2].min():.3f}, {camera_positions[:, 2].max():.3f}]")
    
    plt.tight_layout()
    plt.show()
    
    return test_configs

def test_different_camera_rotations():
    """测试不同的相机旋转模式"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    world_to_camera_matrix = torch.eye(4, device=device)
    
    rotations = ["center_facing", "trajectory_aligned", "no_rotation"]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={'projection': '3d'})
    
    for idx, rotation in enumerate(rotations):
        trajectory = create_spiral_trajectory(
            world_to_camera_matrix,
            center_depth=4.0,
            radius_x=1.0,
            radius_y=0.6,
            radius_z=0.3,
            camera_rotation=rotation,
            n_steps=20,
            device=device,
            num_circles=1.5
        )
        
        trajectory_np = trajectory.cpu().numpy()
        camera_positions = -trajectory_np[:, :3, 3]
        look_at_point = np.array([0.0, 0.0, 4.0])
        
        ax = axes[idx]
        ax.plot(camera_positions[:, 0], camera_positions[:, 1], camera_positions[:, 2], 
                'b-', linewidth=2, label='Camera Path')
        ax.scatter(*look_at_point, color='orange', s=200, marker='*', label='Look-at Target')
        
        # 绘制相机朝向
        for i in range(0, len(camera_positions), 3):
            cam_pos = camera_positions[i]
            if rotation == "center_facing":
                target = look_at_point
            elif rotation == "trajectory_aligned":
                spiral_pos = cam_pos
                target = look_at_point + spiral_pos * 2
            else:  # no_rotation
                spiral_pos = cam_pos
                target = look_at_point + spiral_pos
            
            look_dir = target - cam_pos
            look_dir = look_dir / np.linalg.norm(look_dir) * 0.8
            
            ax.quiver(cam_pos[0], cam_pos[1], cam_pos[2],
                     look_dir[0], look_dir[1], look_dir[2],
                     color='red', alpha=0.7, arrow_length_ratio=0.1)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Camera Rotation: {rotation}')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Testing spiral trajectory generation...")
    configs = test_spiral_trajectory()
    
    print("\n" + "="*50)
    print("Testing different camera rotation modes...")
    test_different_camera_rotations()