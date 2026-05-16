import json
import torch
import os

def inverse_pose(pose: torch.Tensor):
    inv_pose = torch.zeros_like(pose)
    inv_pose[..., :3, :3] = pose[..., :3, :3].transpose(-2, -1)
    inv_pose[..., :3,  3] = -inv_pose[..., :3, :3] @ pose[..., :3, 3]
    inv_pose[..., -1, -1] = 1
    return inv_pose

"""
file_path:
w2c: 4x4
h:
w:
fx:
fy:
cx:
cy

"""


def read_scenes_from_file(file_path: str):
    with open(file_path, 'r') as f:
        scene_paths = f.read().splitlines()
    return scene_paths

root = "exps/dl3dv_ff/out_active"
camera_type = "cameras_difix"
scene_list = os.listdir(root)
record = []
no_found = 0
finish = 0
for scene in scene_list:
    if "gs_nvs" not in scene:
        continue
    scene_id = scene.split("_")[3]
    save_path = f"{root}/{scene}/renders/{camera_type}.json"
    cameras_path = f"{root}/{scene}/renders/{camera_type}.pt"
    if os.path.exists(cameras_path):
        if os.path.exists(save_path):
            finish += 1
            continue
        fv_frames = torch.load(cameras_path, weights_only=False, map_location='cpu')
        out = {}
        for frame in fv_frames:
            new_frame = {
                "file_path": frame.image_path,
                "h": frame.height,
                "w": frame.width,
                "fx": frame.fx.item(),
                "fy": frame.fy.item(),
                "cx": frame.cx.item(),
                "cy": frame.cy.item(),
                "transform_matrix": frame.cam_to_world.numpy().tolist()
            }
            out[frame.image_index] = new_frame
        json.dump(out, open(save_path,"w"))
        finish += 1
    else:
        no_found += 1
        print("No find camera: ", scene)
print(f"{finish} finish, {no_found} no found")

