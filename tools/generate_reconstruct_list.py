import os
import pandas as pd
import os
import shutil
from pathlib import Path
from tqdm import tqdm


def read_scenes_from_file(file_path: str):
    scenes = []
    scene_dict = {}
    with open(file_path, 'r') as f:
        scene_paths = f.read().splitlines()
    
    for scene_path in scene_paths:
        line = scene_path.strip().split('/')
        scenes.append((line[-1], line[-2]))
        scene_dict[line[-1]] = scene_path
    return scenes, scene_dict


def count_scenes_cleanup_failure(root_dir):
    root_path = Path(root_dir)
    TARGET_FILE = 'model/model_030000.pth'
    
    subset_dirs = [d for d in root_path.iterdir() if d.is_dir()]
    
    print(f"check path: {root_path.resolve()}")
    count = 0
    found_scenes = []
    failure = 0

    for scene_path in tqdm(subset_dirs, desc="Processing Subsets"):
        if "logs" in str(scene_path):
            continue
        if scene_path.is_dir():
            target_file_path = scene_path / TARGET_FILE
            expname = str(scene_path).split('/')[-1]
            if not target_file_path.exists():
                print(f"{scene_path.resolve()}")
                # shutil.rmtree(scene_path.resolve())
                failure += 1
            else:
                found_scenes.append(expname)
    
    count = len(found_scenes)
    print(f"Find {count} scenes. {failure} scenes failure.")

def filter_results(csv_file, target_dir):
    forget_results = []
    _, scene_dict = read_scenes_from_file(csv_file)
    current_results = os.listdir(target_dir)
    current_results = [file.split("_")[3] for file in current_results if not "logs" in file]
    
    for scene, path in scene_dict.items():
        if scene not in current_results:
            forget_results.append(path)

    with open("failed_list.txt", 'w', encoding='utf-8') as f:
        for item in forget_results:
            f.write(str(item) + '\n')

def create_trainlist(root_dir, csv_file, save_root=''):
    root_path = Path(root_dir)
    TARGET_FILE = 'renders/cameras.json'
    _, scene_dict = read_scenes_from_file(csv_file)
    
    subset_dirs = [d for d in root_path.iterdir() if d.is_dir()]
    
    print(f"check path: {root_path.resolve()}")
    count = 0
    found_scenes = []
    failure = 0

    for scene_path in tqdm(subset_dirs, desc="Processing sampled renderings"):
        if "logs" in str(scene_path):
            continue
        if scene_path.is_dir():
            target_file_path = scene_path / TARGET_FILE
            target_file_path2 = scene_path / 'renders/view_graph.json'
            scene = str(scene_path).split('/')[-1].split('_')[3]
            if scene not in scene_dict.keys():
                continue
            if not target_file_path.exists() or not target_file_path2.exists():
                # print(f"{scene_path.resolve()}")
                # shutil.rmtree(scene_path.resolve())
                failure += 1
            else:
                found_scenes.append(scene_dict[scene])
    
    count = len(found_scenes)
    print(f"Find {count} scenes. {failure} scenes failure.")

    with open(os.path.join(save_root, "dl3dv_rebuttal_exp1.txt"), 'w', encoding='utf-8') as f:
        for item in found_scenes:
            f.write(str(item) + '\n')
            

csv_file = "/data/dl3dv/dl3dv_ff.txt"
create_trainlist('exps/dl3dv_ff/out_active',
                 csv_file,
                 save_root='/data/dl3dv/')