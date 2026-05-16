
import random
import os
import numpy as np
import PIL
import torch
from torch.utils.data import Dataset
import json
import torch.nn.functional as F
from collections import deque
import re


class DL3DV(Dataset):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.current_iteration = 0

        self.IMAGE_FOLDER_NAME = "images_4" # since you may change to other resolution etc, 4 -> 960P

        try:
            with open(self.config.training.dataset_path, 'r') as f:
                self.all_scenes = f.read().splitlines()
            self.all_scenes = [path for path in self.all_scenes if path.strip()]
            if torch.distributed.get_rank() == 0:
                print(f"Loaded {len(self.all_scenes)} scene paths from {self.config.training.dataset_path}")
        except Exception as e:
            print(f"Error reading dataset paths from '{self.config.training.dataset_path}'")
            raise e
    
        self.inference = self.config.inference.get("if_inference", False)
        # Load file that specifies the input and target view indices to use for inference
        if self.inference:
            self.view_idx_list = dict()
            if self.config.inference.get("view_idx_file_path", None) is not None:
                if os.path.exists(self.config.inference.view_idx_file_path):
                    with open(self.config.inference.view_idx_file_path, 'r') as f:
                        self.view_idx_list = json.load(f)
                        # filter out None values, i.e. scenes that don't have specified input and targetviews
                        self.view_idx_list_filtered = [k for k, v in self.view_idx_list.items() if v is not None]
                    filtered_scene_paths = []
                    for scene in self.all_scenes:
                        scene_name = scene.split('/')[-1]
                        if scene_name in self.view_idx_list_filtered:
                            filtered_scene_paths.append(scene)

                    self.all_scene_paths = filtered_scene_paths

    def __len__(self):
        return len(self.all_scenes)

    def preprocess_frames(self, frames_chosen, image_paths_chosen):
        resize_h = self.config.model.image_tokenizer.image_size
        patch_size = self.config.model.image_tokenizer.patch_size
        square_crop = self.config.training.get("square_crop", False)

        images = []
        intrinsics = []
        for cur_frame, cur_image_path in zip(frames_chosen, image_paths_chosen):
            image = PIL.Image.open(cur_image_path)
            original_image_w, original_image_h = image.size
            
            resize_w = int(resize_h / original_image_h * original_image_w)
            resize_w = int(round(resize_w / patch_size) * patch_size)

            try:
                image = image.resize((resize_w, resize_h), resample=PIL.Image.LANCZOS)
            except Exception as e:
                print(f"Error resizing image {cur_image_path} to {(resize_w, resize_h)}: {e}")
                return None, None, None
            if square_crop:
                min_size = min(resize_h, resize_w)
                start_h = (resize_h - min_size) // 2
                start_w = (resize_w - min_size) // 2
                image = image.crop((start_w, start_h, start_w + min_size, start_h + min_size))

            image = np.array(image) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1).float()
            fxfycxcy = np.array(cur_frame["fxfycxcy"])
            resize_ratio_x = resize_w / original_image_w
            resize_ratio_y = resize_h / original_image_h
            fxfycxcy *= (resize_ratio_x, resize_ratio_y, resize_ratio_x, resize_ratio_y)
            if square_crop:
                fxfycxcy[2] -= start_w
                fxfycxcy[3] -= start_h
            fxfycxcy = torch.from_numpy(fxfycxcy).float()
            images.append(image)
            intrinsics.append(fxfycxcy)

        images = torch.stack(images, dim=0)
        intrinsics = torch.stack(intrinsics, dim=0)
        c2ws = np.stack([np.array(frame["transform_matrix"]) for frame in frames_chosen]) # (num_frames, 4, 4)
        c2ws = torch.from_numpy(c2ws).float()
        return images, intrinsics, c2ws

    def preprocess_poses(self, in_c2ws: torch.Tensor, scene_scale_factor=1.1):
        center = in_c2ws[:, :3, 3].mean(0)
        avg_forward = F.normalize(in_c2ws[:, :3, 2].mean(0), dim=-1) # average forward direction (z of opencv camera)
        avg_down = in_c2ws[:, :3, 1].mean(0) # average down direction (y of opencv camera)
        avg_right = F.normalize(torch.cross(avg_down, avg_forward, dim=-1), dim=-1) # (x of opencv camera)
        avg_down = F.normalize(torch.cross(avg_forward, avg_right, dim=-1), dim=-1) # (y of opencv camera)

        avg_pose = torch.eye(4, device=in_c2ws.device) # average c2w matrix
        avg_pose[:3, :3] = torch.stack([avg_right, avg_down, avg_forward], dim=-1)
        avg_pose[:3, 3] = center 
        avg_pose = torch.linalg.inv(avg_pose) # average w2c matrix
        in_c2ws = avg_pose @ in_c2ws 

        # Rescale the whole scene to a fixed scale
        scene_scale = torch.max(torch.abs(in_c2ws[:, :3, 3]))
        scene_scale = scene_scale_factor * scene_scale

        in_c2ws[:, :3, 3] /= scene_scale

        return in_c2ws

    def view_selector(self, frames):
        if len(frames) < self.config.training.num_views:
            return None

        # sample view candidates
        view_selector_config = self.config.training.view_selector
        
        num_train = len(frames)

        min_frame_dist = view_selector_config.get("min_frame_dist", 25)
        max_frame_dist = min(num_train - 1, view_selector_config.get("max_frame_dist", 100))
        if max_frame_dist <= min_frame_dist:
            return None
        frame_dist = random.randint(min_frame_dist, max_frame_dist)

        if len(frames) <= frame_dist:
            return None
        start_frame = random.randint(0, num_train - frame_dist - 1)
        end_frame = start_frame + frame_dist
        sampled_frames = random.sample(range(start_frame + 1, end_frame), self.config.training.num_views-2)
        image_indices = [start_frame, end_frame] + sampled_frames

        return image_indices

    def __getitem__(self, idx):
        if idx >= len(self.all_scenes):
            return self.__getitem__(random.randint(0, len(self) - 1))
            
        scene = self.all_scenes[idx].strip()
        # transforms.json inside DL3DV
        data_json = json.load(open(os.path.join(scene,'transforms.json'), 'r'))
        frames = data_json["frames"]

        scene_name = scene.split("/")[-1]
        if self.inference and scene_name in self.view_idx_list:
            current_view_idx = self.view_idx_list[scene_name]
            image_indices = current_view_idx["context"] + current_view_idx["target"]
        else:
            image_indices= self.view_selector(frames)
            if image_indices is None:
                return self.__getitem__(random.randint(0, len(self) - 1))
        
        image_paths_chosen = [os.path.join(scene, self.IMAGE_FOLDER_NAME, frames[ic]["file_path"].split('/')[-1]) for ic in image_indices]
        # check if every image path exists
        broken = False
        for p in image_paths_chosen:
            if not os.path.isfile(p):
                broken = True
        if broken or len(image_paths_chosen) != self.config.training.num_views:
            print(f"Broken scene: {scene}, missing image files ({broken}) or invalid view count ({len(image_paths_chosen)} / {self.config.training.num_views}), skip it.")
            self.all_scenes.pop(idx) # remove broken scene to avoid future problems
            return self.__getitem__(random.randint(0, len(self) - 1))
        
        fx, fy, cx, cy, k1, k2, p1, p2 = (data_json[k] for k in ["fl_x", "fl_y", "cx", "cy", "k1", "k2", "p1", "p2"])
        frames_chosen = [{'fxfycxcy': [fx, fy, cx, cy], 'transform_matrix': frames[ic]["transform_matrix"]} for ic in image_indices]
        input_images, input_intrinsics, input_c2ws = self.preprocess_frames(frames_chosen, image_paths_chosen)
        # FIXME: check if input_images shape is correct. HERE I hard code for square image!!!
        if input_images is None:
            return self.__getitem__(random.randint(0, len(self) - 1))
        if input_images.shape[0] != self.config.training.num_views or \
            input_images.shape[1] != 3 or input_images.shape[2] != self.config.model.image_tokenizer.image_size or \
            input_images.shape[3] != self.config.model.image_tokenizer.image_size:
            print(f"Broken scene: {scene}, invalid image shape {input_images.shape}, skip it.")
            self.all_scenes.pop(idx) # remove broken scene to avoid future problems
            return self.__getitem__(random.randint(0, len(self) - 1))

        scene_scale_factor = self.config.training.get("scene_scale_factor", 1.35)
        input_c2ws = self.preprocess_poses(input_c2ws, scene_scale_factor)

        image_indices = torch.tensor(image_indices).long().unsqueeze(-1)  # [v, 1]
        scene_indices = torch.full_like(image_indices, idx)  # [v, 1]
        indices = torch.cat([image_indices, scene_indices], dim=-1)  # [v, 2]
        is_novel = torch.zeros(len(scene_indices)).float().unsqueeze(-1)
        # print(f"scene_path: {scene}, image_shape: {input_images.shape}, intrinsics shape: {input_intrinsics.shape}, c2ws shape: {input_c2ws.shape},  index: {indices}")
        # image_shape here: [T, 3, H, W]
        return {
            "image": input_images,
            "c2w": input_c2ws,
            "fxfycxcy": input_intrinsics,
            "index": indices,
            "scene_name": scene,
            "is_novel": is_novel
        }
    
class DL3DV_test(DL3DV):
    def __len__(self):
        return len(self.all_scene_paths)

    def __getitem__(self, idx):
        if idx >= len(self.all_scene_paths):
            return self.__getitem__(random.randint(0, len(self) - 1))
            
        scene = self.all_scene_paths[idx].strip()
        # transforms.json inside DL3DV
        data_json = json.load(open(os.path.join(scene,'transforms.json'), 'r'))
        frames = data_json["frames"]

        scene_name = scene.split("/")[-1]
        if self.inference and scene_name in self.view_idx_list:
            current_view_idx = self.view_idx_list[scene_name]
            image_indices = current_view_idx["context"] + current_view_idx["target"]
        else:
            image_indices= self.view_selector(frames)
            if image_indices is None:
                return self.__getitem__(random.randint(0, len(self) - 1))
        
        image_paths_chosen = [os.path.join(scene, self.IMAGE_FOLDER_NAME, frames[ic]["file_path"].split('/')[-1]) for ic in image_indices]
        # check if every image path exists
        broken = False
        for p in image_paths_chosen:
            if not os.path.isfile(p):
                broken = True
        if broken or len(image_paths_chosen) != self.config.training.num_views:
            print(f"Broken scene: {scene}, missing image files ({broken}) or invalid view count ({len(image_paths_chosen)} / {self.config.training.num_views}), skip it.")
            self.all_scene_paths.pop(idx) # remove broken scene to avoid future problems
            return self.__getitem__(random.randint(0, len(self) - 1))
        
        fx, fy, cx, cy, k1, k2, p1, p2 = (data_json[k] for k in ["fl_x", "fl_y", "cx", "cy", "k1", "k2", "p1", "p2"])
        frames_chosen = [{'fxfycxcy': [fx, fy, cx, cy], 'transform_matrix': frames[ic]["transform_matrix"]} for ic in image_indices]
        input_images, input_intrinsics, input_c2ws = self.preprocess_frames(frames_chosen, image_paths_chosen)
        # FIXME: check if input_images shape is correct. HERE I hard code for square image!!!
        if input_images is None:
            return self.__getitem__(random.randint(0, len(self) - 1))
        if input_images.shape[0] != self.config.training.num_views or \
            input_images.shape[1] != 3 or input_images.shape[2] != self.config.model.image_tokenizer.image_size or \
            input_images.shape[3] != self.config.model.image_tokenizer.image_size:
            print(f"Broken scene: {scene}, invalid image shape {input_images.shape}, skip it.")
            self.all_scene_paths.pop(idx) # remove broken scene to avoid future problems
            return self.__getitem__(random.randint(0, len(self) - 1))

        scene_scale_factor = self.config.training.get("scene_scale_factor", 1.35)
        input_c2ws = self.preprocess_poses(input_c2ws, scene_scale_factor)

        image_indices = torch.tensor(image_indices).long().unsqueeze(-1)  # [v, 1]
        scene_indices = torch.full_like(image_indices, idx)  # [v, 1]
        indices = torch.cat([image_indices, scene_indices], dim=-1)  # [v, 2]
        is_novel = torch.zeros(len(scene_indices)).float().unsqueeze(-1)
        # print(f"scene_path: {scene}, image_shape: {input_images.shape}, intrinsics shape: {input_intrinsics.shape}, c2ws shape: {input_c2ws.shape},  index: {indices}")
        # image_shape here: [T, 3, H, W]
        return {
            "image": input_images,
            "c2w": input_c2ws,
            "fxfycxcy": input_intrinsics,
            "index": indices,
            "scene_name": scene,
            "is_novel": is_novel
        }


class DL3DV_VG(Dataset):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.current_iteration = 0

        self.freeview_dir = self.config.training.view_selector.freeview_dir
        self.IMAGE_FOLDER_NAME = "images_4" # since you may change to other resolution etc, 4 -> 960P

        try:
            with open(self.config.training.dataset_path, 'r') as f:
                self.all_scenes = f.read().splitlines()
            self.all_scenes = [path for path in self.all_scenes if path.strip()]
            if torch.distributed.get_rank() == 0:
                print(f"Loaded {len(self.all_scenes)} scene paths from {self.config.training.dataset_path}")
        except Exception as e:
            print(f"Error reading dataset paths from '{self.config.training.dataset_path}'")
            raise e
        
        self.inference = self.config.inference.get("if_inference", False)

    def __len__(self):
        return len(self.all_scenes)

    def preprocess_frames(self, frames_chosen, image_paths_chosen):
        resize_h = self.config.model.image_tokenizer.image_size
        patch_size = self.config.model.image_tokenizer.patch_size
        square_crop = self.config.training.get("square_crop", False)

        images = []
        intrinsics = []
        for cur_frame, cur_image_path in zip(frames_chosen, image_paths_chosen):
            image = PIL.Image.open(cur_image_path)
            original_image_w, original_image_h = image.size
            
            resize_w = int(resize_h / original_image_h * original_image_w)
            resize_w = int(round(resize_w / patch_size) * patch_size)

            try:
                image = image.resize((resize_w, resize_h), resample=PIL.Image.LANCZOS)
            except Exception as e:
                print(f"Error resizing image {cur_image_path} to {(resize_w, resize_h)}: {e}")
                return None, None, None
            if square_crop:
                min_size = min(resize_h, resize_w)
                start_h = (resize_h - min_size) // 2
                start_w = (resize_w - min_size) // 2
                image = image.crop((start_w, start_h, start_w + min_size, start_h + min_size))

            image = np.array(image) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1).float()
            fxfycxcy = np.array(cur_frame["fxfycxcy"])
            resize_ratio_x = resize_w / original_image_w
            resize_ratio_y = resize_h / original_image_h
            fxfycxcy *= (resize_ratio_x, resize_ratio_y, resize_ratio_x, resize_ratio_y)
            if square_crop:
                fxfycxcy[2] -= start_w
                fxfycxcy[3] -= start_h
            fxfycxcy = torch.from_numpy(fxfycxcy).float()
            images.append(image)
            intrinsics.append(fxfycxcy)

        images = torch.stack(images, dim=0)
        intrinsics = torch.stack(intrinsics, dim=0)
        c2ws = np.stack([np.array(frame["transform_matrix"]) for frame in frames_chosen]) # (num_frames, 4, 4)
        c2ws = torch.from_numpy(c2ws).float()
        return images, intrinsics, c2ws

    def preprocess_poses(self, in_c2ws: torch.Tensor, scene_scale_factor=1.1):
        center = in_c2ws[:, :3, 3].mean(0)
        avg_forward = F.normalize(in_c2ws[:, :3, 2].mean(0), dim=-1) # average forward direction (z of opencv camera)
        avg_down = in_c2ws[:, :3, 1].mean(0) # average down direction (y of opencv camera)
        avg_right = F.normalize(torch.cross(avg_down, avg_forward, dim=-1), dim=-1) # (x of opencv camera)
        avg_down = F.normalize(torch.cross(avg_forward, avg_right, dim=-1), dim=-1) # (y of opencv camera)

        avg_pose = torch.eye(4, device=in_c2ws.device) # average c2w matrix
        avg_pose[:3, :3] = torch.stack([avg_right, avg_down, avg_forward], dim=-1)
        avg_pose[:3, 3] = center 
        avg_pose = torch.linalg.inv(avg_pose) # average w2c matrix
        in_c2ws = avg_pose @ in_c2ws 

        # Rescale the whole scene to a fixed scale
        scene_scale = torch.max(torch.abs(in_c2ws[:, :3, 3]))
        scene_scale = scene_scale_factor * scene_scale

        in_c2ws[:, :3, 3] /= scene_scale

        return in_c2ws
    
    def view_selector(self, frames, view_graph= None):
        if len(frames) < self.config.training.num_views:
            return None
        # sample view candidates
        view_selector_config = self.config.training.view_selector 
        num_train = int(view_selector_config.get("train_frac", 1) * len(frames))

        min_frame_dist = view_selector_config.get("min_frame_dist", 25)
        max_frame_dist = min(num_train - 1, view_selector_config.get("max_frame_dist", 100))

        # curriculum learning
        curriculum_max_iter = view_selector_config.get('curriculum_iter', 3000)
        progress = min(self.current_iteration / curriculum_max_iter, 1.0)
        min_frame_dist_start = view_selector_config.get("curriculum_start_min_frame_dist", 48)
        max_frame_dist_start = view_selector_config.get("curriculum_start_max_frame_dist", 64)
        cur_min_frame_dist = int(min_frame_dist_start + (min_frame_dist - min_frame_dist_start) * progress)
        cur_max_frame_dist = int(max_frame_dist_start + (max_frame_dist - max_frame_dist_start) * progress)

        if cur_max_frame_dist <= cur_min_frame_dist:
            return None
        frame_dist = random.randint(cur_min_frame_dist, cur_max_frame_dist)

        view_selector_config = self.config.training.view_selector
        if view_graph is not None and random.random() > 0.5:
            node_weights_sum = {nk: sum(item[1] for item in nv) for nk, nv in view_graph.items()}

            available_nodes = {
                nk: weight_sum 
                for nk, weight_sum in node_weights_sum.items() 
                if len(view_graph.get(nk, [])) >= self.config.training.num_views - 1
            }
            nk_sort_list = sorted(
                available_nodes.keys(), 
                key=lambda nk: available_nodes[nk], 
                reverse=True
            )
            
            TOP_PERCENTAGE = frame_dist / max_frame_dist
            k = max(1, int(len(nk_sort_list) * TOP_PERCENTAGE))
            
            top_k_nodes = nk_sort_list[:k]
            if len(top_k_nodes) > 0:
                start_frame = random.choice(top_k_nodes)
                neighbors = sorted(view_graph[start_frame], key=lambda item: item[1])
                neighbors_index = [str(n[0]) for n in neighbors]

                sampled_frames = random.sample(neighbors_index, self.config.training.num_views - 1)
                image_indices = [start_frame] + sampled_frames 
                if all(s.startswith('fv') for s in image_indices):
                    targe_frame = find_first_valid_id(neighbors_index, view_graph, k=2, N=random.randint(0, num_train - 1))
                    image_indices[-1] = targe_frame
                sort_key = lambda s: 0 if str(s).startswith('fv_') else 1
                image_indices = sorted(image_indices, key=sort_key)
            else:
                if len(frames) <= frame_dist:
                    return None
                start_frame = random.randint(0, num_train - frame_dist - 1)
                end_frame = start_frame + frame_dist
                sampled_frames = random.sample(range(start_frame + 1, end_frame), self.config.training.num_views-2)
                image_indices = [start_frame, end_frame] + sampled_frames
        else:
            if len(frames) <= frame_dist:
                return None
            start_frame = random.randint(0, num_train - frame_dist - 1)
            end_frame = start_frame + frame_dist
            sampled_frames = random.sample(range(start_frame + 1, end_frame), self.config.training.num_views-2)
            image_indices = [start_frame, end_frame] + sampled_frames

        a = image_indices[:-2]
        b = image_indices[-2:]
        random.shuffle(a)

        return a + b

    def __getitem__(self, idx):
        if idx >= len(self.all_scenes):
            return self.__getitem__(random.randint(0, len(self) - 1))
            
        scene = self.all_scenes[idx].strip()
        # transforms.json inside DL3DV
        data_json = json.load(open(os.path.join(scene,'transforms.json'), 'r'))
        frames = data_json["frames"]
        scene_id = scene.split("/")[-1]

        # get freeviews
        cameras_path = f"{self.freeview_dir}/gs_nvs_DL3DV10K_{scene_id}_3dgs/renders/cameras_difix.json"
        view_graph_path = f"{self.freeview_dir}/gs_nvs_DL3DV10K_{scene_id}_3dgs/renders/view_graph.json"
        if os.path.exists(cameras_path) and os.path.exists(view_graph_path):
            fv_frames = json.load(open(cameras_path, "r"))
            view_graph = json.load(open(view_graph_path, "r"))
            image_idx_to_index = {
                f: i for i, f in enumerate(fv_frames.keys())
            }
            fv_frames = list(fv_frames.values())
        else:
            view_graph = None
            fv_frames = []
            image_idx_to_index = {}

        train_ind = [int(ind) for ind in view_graph.keys() if 'fv' not in str(ind)]
        assert max(train_ind) <= len(train_ind), "scene_id"
        image_indices = self.view_selector(frames, view_graph)

        if image_indices is None:
            return self.__getitem__(random.randint(0, len(self) - 1))
        
        image_paths_chosen = []
        frames_chosen = []
        image_indices_final = []
        is_novel = []
        fx, fy, cx, cy, k1, k2, p1, p2 = (data_json[k] for k in ["fl_x", "fl_y", "cx", "cy", "k1", "k2", "p1", "p2"])
        for ic in image_indices:
            if str(ic).find('fv') < 0:
                image_paths_chosen.append(os.path.join(scene, self.IMAGE_FOLDER_NAME, frames[int(ic)]["file_path"].split('/')[-1]))
                frames_chosen.append({'fxfycxcy': [fx, fy, cx, cy], 'transform_matrix': frames[int(ic)]["transform_matrix"]})
                image_indices_final.append(int(ic))
                is_novel.append(0)
            else:
                index = image_idx_to_index[ic]
                img_path = fv_frames[index]["image_path"]
                img_path = replace_difix_with_choices(img_path)
                image_paths_chosen.append(img_path)
                frames_chosen.append({'fxfycxcy': [fx, fy, cx, cy], 'transform_matrix': fv_frames[index]["transform_matrix"]})
                image_indices_final.append(index + len(frames))
                is_novel.append(1)

        # check if every image path exists
        broken = False
        for p in image_paths_chosen:
            if not os.path.isfile(p):
                broken = True
        if broken or len(image_paths_chosen) != self.config.training.num_views:
            print(f"Broken scene: {scene}, missing image files ({broken}) or invalid view count ({len(image_paths_chosen)} / {self.config.training.num_views}), skip it.")
            self.all_scenes.pop(idx) # remove broken scene to avoid future problems
            return self.__getitem__(random.randint(0, len(self) - 1))

        input_images, input_intrinsics, input_c2ws = self.preprocess_frames(frames_chosen, image_paths_chosen)

        # FIXME: check if input_images shape is correct. HERE I hard code for square image!!!
        if input_images is None:
            return self.__getitem__(random.randint(0, len(self) - 1))
        if input_images.shape[0] != self.config.training.num_views or \
            input_images.shape[1] != 3 or input_images.shape[2] != self.config.model.image_tokenizer.image_size or \
            input_images.shape[3] != self.config.model.image_tokenizer.image_size:
            print(f"Broken scene: {scene}, invalid image shape {input_images.shape}, skip it.")
            self.all_scenes.pop(idx) # remove broken scene to avoid future problems
            return self.__getitem__(random.randint(0, len(self) - 1))

        scene_scale_factor = self.config.training.get("scene_scale_factor", 1.35)
        input_c2ws = self.preprocess_poses(input_c2ws, scene_scale_factor)

        image_indices = torch.tensor(image_indices_final).long().unsqueeze(-1)  # [v, 1]
        scene_indices = torch.full_like(image_indices, idx)  # [v, 1]
        indices = torch.cat([image_indices, scene_indices], dim=-1)  # [v, 2]
        is_novel = torch.tensor(is_novel).float().unsqueeze(-1)  # [v, 1]
        # print(f"scene_path: {scene}, image_shape: {input_images.shape}, intrinsics shape: {input_intrinsics.shape}, c2ws shape: {input_c2ws.shape},  index: {indices}")
        # image_shape here: [T, 3, H, W]
        return {
            "image": input_images,
            "c2w": input_c2ws,
            "fxfycxcy": input_intrinsics,
            "index": indices,
            "scene_name": scene_id,
            "is_novel": is_novel
        }


class MipNeRFDataset(Dataset):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.IMAGE_FOLDER_NAME = "images_4"

        try:
            with open(self.config.training.dataset_path, 'r') as f:
                self.all_scene_paths = f.read().splitlines()
            self.all_scene_paths = [path for path in self.all_scene_paths if path.strip()]
            if torch.distributed.get_rank() == 0:
                print(f"Loaded {len(self.all_scene_paths)} scene paths from {self.config.training.dataset_path}")
        except Exception as e:
            print(f"Error reading dataset paths from '{self.config.training.dataset_path}'")
            raise e
        

        self.inference = self.config.inference.get("if_inference", False)
        # Load file that specifies the input and target view indices to use for inference
        if self.inference:
            self.view_idx_list = dict()
            if self.config.inference.get("view_idx_file_path", None) is not None:
                if os.path.exists(self.config.inference.view_idx_file_path):
                    with open(self.config.inference.view_idx_file_path, 'r') as f:
                        self.view_idx_list = json.load(f)
                        # filter out None values, i.e. scenes that don't have specified input and targetviews
                        self.view_idx_list_filtered = [k for k, v in self.view_idx_list.items() if v is not None]
                    filtered_scene_paths = []
                    for scene in self.all_scene_paths:
                        scene_name = scene.split("/")[-1]
                        if scene_name in self.view_idx_list_filtered:
                            filtered_scene_paths.append(scene)

                    self.all_scene_paths = filtered_scene_paths
    
    def __len__(self):
        return len(self.all_scene_paths)

    def preprocess_frames(self, frames_chosen, image_paths_chosen):
        resize_h = self.config.model.image_tokenizer.image_size
        patch_size = self.config.model.image_tokenizer.patch_size
        square_crop = self.config.training.get("square_crop", False)

        images = []
        intrinsics = []
        for cur_frame, cur_image_path in zip(frames_chosen, image_paths_chosen):
            image = PIL.Image.open(cur_image_path)
            original_image_w, original_image_h = image.size
            
            resize_w = int(resize_h / original_image_h * original_image_w)
            resize_w = int(round(resize_w / patch_size) * patch_size)

            try:
                image = image.resize((resize_w, resize_h), resample=PIL.Image.LANCZOS)
            except Exception as e:
                print(f"Error resizing image {cur_image_path} to {(resize_w, resize_h)}: {e}")
                return None, None, None
            if square_crop:
                min_size = min(resize_h, resize_w)
                start_h = (resize_h - min_size) // 2
                start_w = (resize_w - min_size) // 2
                image = image.crop((start_w, start_h, start_w + min_size, start_h + min_size))

            image = np.array(image) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1).float()
            fxfycxcy = np.array(cur_frame["fxfycxcy"])
            resize_ratio_x = resize_w / original_image_w
            resize_ratio_y = resize_h / original_image_h
            fxfycxcy *= (resize_ratio_x, resize_ratio_y, resize_ratio_x, resize_ratio_y)
            if square_crop:
                fxfycxcy[2] -= start_w
                fxfycxcy[3] -= start_h
            fxfycxcy = torch.from_numpy(fxfycxcy).float()
            images.append(image)
            intrinsics.append(fxfycxcy)

        images = torch.stack(images, dim=0)
        intrinsics = torch.stack(intrinsics, dim=0)
        c2ws = np.stack([np.array(frame["transform_matrix"]) for frame in frames_chosen]) # (num_frames, 4, 4)
        c2ws = torch.from_numpy(c2ws).float()
        return images, intrinsics, c2ws

    def preprocess_poses(self, in_c2ws: torch.Tensor, scene_scale_factor=1.1):
        center = in_c2ws[:, :3, 3].mean(0)
        avg_forward = F.normalize(in_c2ws[:, :3, 2].mean(0), dim=-1) # average forward direction (z of opencv camera)
        avg_down = in_c2ws[:, :3, 1].mean(0) # average down direction (y of opencv camera)
        avg_right = F.normalize(torch.cross(avg_down, avg_forward, dim=-1), dim=-1) # (x of opencv camera)
        avg_down = F.normalize(torch.cross(avg_forward, avg_right, dim=-1), dim=-1) # (y of opencv camera)

        avg_pose = torch.eye(4, device=in_c2ws.device) # average c2w matrix
        avg_pose[:3, :3] = torch.stack([avg_right, avg_down, avg_forward], dim=-1)
        avg_pose[:3, 3] = center 
        avg_pose = torch.linalg.inv(avg_pose) # average w2c matrix
        in_c2ws = avg_pose @ in_c2ws 

        # Rescale the whole scene to a fixed scale
        scene_scale = torch.max(torch.abs(in_c2ws[:, :3, 3]))
        scene_scale = scene_scale_factor * scene_scale

        in_c2ws[:, :3, 3] /= scene_scale

        return in_c2ws

    def view_selector(self, frames):
        return None

    def __getitem__(self, idx):
        if idx >= len(self.all_scene_paths):
            return self.__getitem__(random.randint(0, len(self) - 1))
            
        scene = self.all_scene_paths[idx].strip()
        # transforms.json inside DL3DV
        data_json = json.load(open(os.path.join(scene,'transforms.json'), 'r'))
        frames = data_json["frames"]

        scene_name = scene.split("/")[-1]
        if self.inference and scene_name in self.view_idx_list:
            current_view_idx = self.view_idx_list[scene_name]
            image_indices = current_view_idx["context"] + current_view_idx["target"]
        else:
            image_indices= self.view_selector(frames)
            if image_indices is None:
                return self.__getitem__(random.randint(0, len(self) - 1))
        
        image_paths_chosen = [os.path.join(scene, self.IMAGE_FOLDER_NAME, frames[ic]["file_path"].split('/')[-1]) for ic in image_indices]
        # check if every image path exists
        broken = False
        for p in image_paths_chosen:
            if not os.path.isfile(p):
                broken = True

        if broken or len(image_paths_chosen) != self.config.training.num_views:
            print(f"Broken scene: {scene}, missing image files ({broken}) or invalid view count ({len(image_paths_chosen)} / {self.config.training.num_views}), skip it.")
            self.all_scene_paths.pop(idx) # remove broken scene to avoid future problems
            return self.__getitem__(random.randint(0, len(self) - 1))
        
        fx, fy, cx, cy, k1, k2, p1, p2 = (data_json[k] for k in ["fl_x", "fl_y", "cx", "cy", "k1", "k2", "p1", "p2"])
        frames_chosen = [{'fxfycxcy': [fx, fy, cx, cy], 'transform_matrix': frames[ic]["transform_matrix"]} for ic in image_indices]
        input_images, input_intrinsics, input_c2ws = self.preprocess_frames(frames_chosen, image_paths_chosen)
        # FIXME: check if input_images shape is correct. HERE I hard code for square image!!!
        if input_images is None:
            return self.__getitem__(random.randint(0, len(self) - 1))
        if input_images.shape[0] != self.config.training.num_views or \
            input_images.shape[1] != 3 or input_images.shape[2] != self.config.model.image_tokenizer.image_size or \
            input_images.shape[3] != self.config.model.image_tokenizer.image_size:
            print(f"Broken scene: {scene}, invalid image shape {input_images.shape}, skip it.")
            self.all_scene_paths.pop(idx) # remove broken scene to avoid future problems
            return self.__getitem__(random.randint(0, len(self) - 1))

        scene_scale_factor = self.config.training.get("scene_scale_factor", 1.35)
        input_c2ws = self.preprocess_poses(input_c2ws, scene_scale_factor)

        image_indices = torch.tensor(image_indices).long().unsqueeze(-1)  # [v, 1]
        scene_indices = torch.full_like(image_indices, idx)  # [v, 1]
        indices = torch.cat([image_indices, scene_indices], dim=-1)  # [v, 2]
        is_novel = torch.zeros(len(scene_indices)).float().unsqueeze(-1)
        # print(f"scene_path: {scene}, image_shape: {input_images.shape}, intrinsics shape: {input_intrinsics.shape}, c2ws shape: {input_c2ws.shape},  index: {indices}")
        # image_shape here: [T, 3, H, W]
        return {
            "image": input_images,
            "c2w": input_c2ws,
            "fxfycxcy": input_intrinsics,
            "index": indices,
            "scene_name": scene,
            "is_novel": is_novel
        }

class TTDataset(Dataset):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.IMAGE_FOLDER_NAME = "images_2"

        try:
            with open(self.config.training.dataset_path, 'r') as f:
                self.all_scene_paths = f.read().splitlines()
            self.all_scene_paths = [path for path in self.all_scene_paths if path.strip()]
            if torch.distributed.get_rank() == 0:
                print(f"Loaded {len(self.all_scene_paths)} scene paths from {self.config.training.dataset_path}")
        except Exception as e:
            print(f"Error reading dataset paths from '{self.config.training.dataset_path}'")
            raise e
        
        self.inference = self.config.inference.get("if_inference", False)
        # Load file that specifies the input and target view indices to use for inference
        if self.inference:
            self.view_idx_list = dict()
            if self.config.inference.get("view_idx_file_path", None) is not None:
                if os.path.exists(self.config.inference.view_idx_file_path):
                    with open(self.config.inference.view_idx_file_path, 'r') as f:
                        self.view_idx_list = json.load(f)
                        # filter out None values, i.e. scenes that don't have specified input and targetviews
                        self.view_idx_list_filtered = [k for k, v in self.view_idx_list.items() if v is not None]
                    filtered_scene_paths = []
                    for scene in self.all_scene_paths:
                        scene_name = scene.split("/")[-1]
                        if scene_name in self.view_idx_list_filtered:
                            filtered_scene_paths.append(scene)

                    self.all_scene_paths = filtered_scene_paths
    
    def __len__(self):
        return len(self.all_scene_paths)

    def preprocess_frames(self, frames_chosen, image_paths_chosen):
        resize_h = self.config.model.image_tokenizer.image_size
        patch_size = self.config.model.image_tokenizer.patch_size
        square_crop = self.config.training.get("square_crop", False)

        images = []
        intrinsics = []
        for cur_frame, cur_image_path in zip(frames_chosen, image_paths_chosen):
            image = PIL.Image.open(cur_image_path)
            original_image_w, original_image_h = image.size
            
            resize_w = int(resize_h / original_image_h * original_image_w)
            resize_w = int(round(resize_w / patch_size) * patch_size)

            try:
                image = image.resize((resize_w, resize_h), resample=PIL.Image.LANCZOS)
            except Exception as e:
                print(f"Error resizing image {cur_image_path} to {(resize_w, resize_h)}: {e}")
                return None, None, None
            if square_crop:
                min_size = min(resize_h, resize_w)
                start_h = (resize_h - min_size) // 2
                start_w = (resize_w - min_size) // 2
                image = image.crop((start_w, start_h, start_w + min_size, start_h + min_size))

            image = np.array(image) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1).float()
            fxfycxcy = np.array(cur_frame["fxfycxcy"])
            resize_ratio_x = resize_w / original_image_w
            resize_ratio_y = resize_h / original_image_h
            fxfycxcy *= (resize_ratio_x, resize_ratio_y, resize_ratio_x, resize_ratio_y)
            if square_crop:
                fxfycxcy[2] -= start_w
                fxfycxcy[3] -= start_h
            fxfycxcy = torch.from_numpy(fxfycxcy).float()
            images.append(image)
            intrinsics.append(fxfycxcy)

        images = torch.stack(images, dim=0)
        intrinsics = torch.stack(intrinsics, dim=0)
        c2ws = np.stack([np.array(frame["transform_matrix"]) for frame in frames_chosen]) # (num_frames, 4, 4)
        c2ws = torch.from_numpy(c2ws).float()
        return images, intrinsics, c2ws

    def preprocess_poses(self, in_c2ws: torch.Tensor, scene_scale_factor=1.1):
        center = in_c2ws[:, :3, 3].mean(0)
        avg_forward = F.normalize(in_c2ws[:, :3, 2].mean(0), dim=-1) # average forward direction (z of opencv camera)
        avg_down = in_c2ws[:, :3, 1].mean(0) # average down direction (y of opencv camera)
        avg_right = F.normalize(torch.cross(avg_down, avg_forward, dim=-1), dim=-1) # (x of opencv camera)
        avg_down = F.normalize(torch.cross(avg_forward, avg_right, dim=-1), dim=-1) # (y of opencv camera)

        avg_pose = torch.eye(4, device=in_c2ws.device) # average c2w matrix
        avg_pose[:3, :3] = torch.stack([avg_right, avg_down, avg_forward], dim=-1)
        avg_pose[:3, 3] = center 
        avg_pose = torch.linalg.inv(avg_pose) # average w2c matrix
        in_c2ws = avg_pose @ in_c2ws 

        # Rescale the whole scene to a fixed scale
        scene_scale = torch.max(torch.abs(in_c2ws[:, :3, 3]))
        scene_scale = scene_scale_factor * scene_scale

        in_c2ws[:, :3, 3] /= scene_scale

        return in_c2ws

    def view_selector(self, frames):
        return None

    def __getitem__(self, idx):
        if idx >= len(self.all_scene_paths):
            return self.__getitem__(random.randint(0, len(self) - 1))
            
        scene = self.all_scene_paths[idx].strip()
        # transforms.json inside DL3DV
        data_json = json.load(open(os.path.join(scene,'transforms.json'), 'r'))
        frames = data_json["frames"]

        scene_name = scene.split("/")[-1]
        if self.inference and scene_name in self.view_idx_list:
            current_view_idx = self.view_idx_list[scene_name]
            image_indices = current_view_idx["context"] + current_view_idx["target"]
        else:
            image_indices= self.view_selector(frames)
            if image_indices is None:
                return self.__getitem__(random.randint(0, len(self) - 1))
        
        image_paths_chosen = [os.path.join(scene, self.IMAGE_FOLDER_NAME, frames[ic]["file_path"].split('/')[-1]) for ic in image_indices]
        # check if every image path exists
        broken = False
        for p in image_paths_chosen:
            if not os.path.isfile(p):
                broken = True

        if broken or len(image_paths_chosen) != self.config.training.num_views:
            print(f"Broken scene: {scene}, missing image files ({broken}) or invalid view count ({len(image_paths_chosen)} / {self.config.training.num_views}), skip it.")
            self.all_scene_paths.pop(idx) # remove broken scene to avoid future problems
            return self.__getitem__(random.randint(0, len(self) - 1))
        
        fx, fy, cx, cy, k1, k2, p1, p2 = (data_json[k] for k in ["fl_x", "fl_y", "cx", "cy", "k1", "k2", "p1", "p2"])
        frames_chosen = [{'fxfycxcy': [fx, fy, cx, cy], 'transform_matrix': frames[ic]["transform_matrix"]} for ic in image_indices]
        input_images, input_intrinsics, input_c2ws = self.preprocess_frames(frames_chosen, image_paths_chosen)
        # FIXME: check if input_images shape is correct. HERE I hard code for square image!!!
        if input_images is None:
            return self.__getitem__(random.randint(0, len(self) - 1))
        if input_images.shape[0] != self.config.training.num_views or \
            input_images.shape[1] != 3 or input_images.shape[2] != self.config.model.image_tokenizer.image_size or \
            input_images.shape[3] != self.config.model.image_tokenizer.image_size:
            print(f"Broken scene: {scene}, invalid image shape {input_images.shape}, skip it.")
            self.all_scene_paths.pop(idx) # remove broken scene to avoid future problems
            return self.__getitem__(random.randint(0, len(self) - 1))

        scene_scale_factor = self.config.training.get("scene_scale_factor", 1.35)
        input_c2ws = self.preprocess_poses(input_c2ws, scene_scale_factor)

        image_indices = torch.tensor(image_indices).long().unsqueeze(-1)  # [v, 1]
        scene_indices = torch.full_like(image_indices, idx)  # [v, 1]
        indices = torch.cat([image_indices, scene_indices], dim=-1)  # [v, 2]
        is_novel = torch.zeros(len(scene_indices)).float().unsqueeze(-1)
        # print(f"scene_path: {scene}, image_shape: {input_images.shape}, intrinsics shape: {input_intrinsics.shape}, c2ws shape: {input_c2ws.shape},  index: {indices}")
        # image_shape here: [T, 3, H, W]
        return {
            "image": input_images,
            "c2w": input_c2ws,
            "fxfycxcy": input_intrinsics,
            "index": indices,
            "scene_name": scene,
            "is_novel": is_novel
        }


def _check_id_expansion(start_id, graph, max_depth, memo):
    if start_id in memo:
        return memo[start_id]
    queue = deque([(start_id, 0)])
    visited = {start_id}

    while queue:
        current_id, depth = queue.popleft()
        if 'fv' not in current_id:
            memo[start_id] = current_id 
            return current_id
        
        if current_id in memo:
            found_id = memo[current_id]
            if found_id is not None:
                memo[start_id] = found_id 
                return found_id
            else:
                continue

        if depth >= max_depth:
            continue

        if current_id in graph:
            for neighbour, _ in graph[current_id]:
                if str(neighbour) not in visited:
                    visited.add(str(neighbour))
                    queue.append((str(neighbour), depth + 1))

    memo[start_id] = None
    return None

def find_first_valid_id(neighbour_list, graph, k, N):
    memo = {}
    for start_id in neighbour_list:
        found_id = _check_id_expansion(str(start_id), graph, k, memo)
        if found_id is not None:
            return found_id
    return random.randint(0, N)

def replace_difix_with_choices(
    text: str, 
    prob_one=0.3, 
    prob_two=0.3
) -> str:

    options = ["difix_1", "difix_2", "difix"]
    weights = [prob_one, prob_two, 1.0 - prob_one - prob_two]

    def replacer(match: re.Match) -> str:
        return random.choices(options, weights=weights, k=1)[0]

    return re.sub(r'difix', replacer, text)