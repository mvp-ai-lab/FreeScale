import os
from pathlib import Path
from PIL import Image
import csv
from typing import List, Tuple
import pandas as pd
from pycolmap import SceneManager

def check_image_integrity(image_dir: str):
    image_path = Path(image_dir)
    
    valid_extensions = ['.jpg', '.jpeg', '.png']
    total_files = 0
    corrupted_files = []
    valid_files = []

    for file_path in image_path.rglob('*'):
        if file_path.suffix.lower() in valid_extensions:
            total_files += 1
            file_name = file_path.relative_to(image_path)
            
            try:
                img = Image.open(file_path)
                img.verify() 
                valid_files.append(file_name)
            except Exception as e:
                print(f"❌ Truncked image: {file_name} {e}")
                corrupted_files.append(file_name)

    # print(f"Total image number: {total_files}, {len(corrupted_files)} invalid.")
    if corrupted_files:
        with open(image_path/"invalid_list.txt", 'w') as f:
            for cf in corrupted_files:
                f.write(f"{cf.resolve()}\n")
    return len(valid_files)

def save_to_csv(data_list: List[Tuple[str, str, int]], output_csv_path: str):
    header = ['scene_id', 'subset', 'num_of_valid']
    file_exists = os.path.exists(output_csv_path)
    try:
        with open(output_csv_path, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)

            if not file_exists:
                writer.writerow(header)

            writer.writerows(data_list)
    except Exception as e:
        print(e)

def create_valid_list(csv_path) -> List[str]:
    df = pd.read_csv(csv_path)
    combined_series = df['scene_id'].astype(str) + '_' + df['subset'].astype(str)
    return combined_series.tolist()
    

data_root = '/cephyr/users/qingwenz/Alvis/data/dl3dv/DL3DV-10K/'
colmap_root = '/cephyr/users/qingwenz/Alvis/data/dl3dv/DL3DV-10K_c2'
csv_path = '/cephyr/users/qingwenz/Alvis/data/dl3dv/DL3DV-valid2.csv'

exist_valid_list = create_valid_list('/cephyr/users/qingwenz/Alvis/data/dl3dv/DL3DV-valid.csv')

for subset in range(1, 11):
    subset = str(subset) + 'K'
    path = os.path.join(data_root, subset)
    valid_scene = []
    missing_paths = []
    for scene in os.listdir(path):
        valid_flag = False
        colmap_dir = os.path.join(colmap_root, subset, scene)

        scene_key = f"{scene}_{subset}"
        if scene_key in exist_valid_list:
            continue

        if os.path.exists(colmap_dir):
            num_valid_imgs = check_image_integrity(os.path.join(data_root, subset, scene))
            if not os.path.exists(os.path.join(data_root, subset, scene, 'transforms.json')):
                if os.path.exists(os.path.join(colmap_dir, 'transforms.json')):
                    # print(f"cp {os.path.join(colmap_dir, 'transforms.json')} {os.path.join(data_root, subset, scene)}")
                    os.system(f"cp {os.path.join(colmap_dir, 'transforms.json')} {os.path.join(data_root, subset, scene)}")
                    valid_flag = True
            else:
                valid_flag = True
            
            if valid_flag:
                if os.path.exists(os.path.join(colmap_dir, 'colmap/sparse/0/images.bin')):
                    manager = SceneManager(os.path.join(colmap_dir, 'colmap/sparse/0/'))
                    manager.load_cameras()
                    manager.load_images()
                    manager.load_points3D()
                    imdata = manager.images

                    try:
                        if len(imdata) != num_valid_imgs:
                            missing_paths = (str(scene), str(subset), num_valid_imgs, len(imdata))
                            valid_flag  = False
                        else:
                            valid_scene.append((scene, subset, num_valid_imgs))
                            # print(f"cp -r {os.path.join(colmap_root, subset, scene, 'colmap/sparse')} {os.path.join(data_root, subset, scene)}")
                            if not os.path.exists(os.path.join(data_root, subset, scene, 'colmap/sparse')):
                                os.system(f"cp -r {os.path.join(colmap_root, subset, scene, 'colmap/sparse')} {os.path.join(data_root, subset, scene)}")
                                os.system(f"rm -r {os.path.join(data_root, subset, scene, 'sparse/0/models')}")
                    except:
                        missing_paths = (str(scene), str(subset), -1, -1)
                        valid_flag  = False

        else:
            # print(f"No found {os.path.join(colmap_root, subset, scene)}")
            missing_paths.append((scene, subset, -1))
        
    save_to_csv(valid_scene, csv_path)
    save_to_csv(missing_paths, csv_path.replace("DL3DV-valid", "invalid"))
    

