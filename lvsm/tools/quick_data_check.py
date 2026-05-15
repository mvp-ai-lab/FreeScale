import os, json

def is_valid_scene(scene_path):
    flag = True
    for cam_name in ['cam1', 'cam2', 'cam3', 'cam4', 'cam5', 'cam6']:
        cam_path = os.path.join(scene_path, cam_name)
        target_image = os.path.join(cam_path, 'rgba_00061.png')
        if not os.path.isdir(cam_path) or not os.path.isfile(target_image):
            # print(os.path.isdir(cam_path), os.path.isfile(target_image))
            flag = False
    return flag

def main(
    dataset_path: str = "/proj/berzelius-2023-364/data/kubric-v2",
):
    for scene_name in os.listdir(dataset_path):
        scene_path = os.path.join(dataset_path, scene_name)
        if not is_valid_scene(scene_path):
            print(f"------ [ERROR] Invalid scene at {scene_path}. Please check the folder structure.")
            # continue
            break

def change_img_name(
    # scene_path: str = "/proj/berzelius-2023-364/data/kubric-v2/scene_00222",
    data_path: str = "/proj/berzelius-2023-364/data/kubric-v2",
):
    for scene_name in os.listdir(data_path):
        scene_path = os.path.join(data_path, scene_name)
        if not is_valid_scene(scene_path):
            print(f"------ [ERROR] Invalid scene at {scene_path}. Please check the folder structure.")
            # continue
        
        # print(f"Processing scene: {scene_name}")
        
        # Rename images in each camera folder
        for cam_name in ['cam1', 'cam2', 'cam3', 'cam4', 'cam5', 'cam6']:
            cam_path = os.path.join(scene_path, cam_name)
            for img_name in os.listdir(cam_path):
                if img_name.startswith('rgba_') and img_name.endswith('.png') and len(img_name.split('_')[-1].split('.')[0]) == 4:
                    new_img_name = img_name.replace('rgba_', 'rgba_0')
                    os.rename(os.path.join(cam_path, img_name), os.path.join(cam_path, new_img_name))
                # print(f"Renamed {img_name} to {new_img_name} in {cam_path}")
if __name__ == "__main__":
    # main()
    change_img_name()