import random
import numpy as np
import torch
import torch.nn as nn
from easydict import EasyDict as edict
from einops import rearrange
import imageio



def create_video_from_frames(frames, output_video_file, framerate=30):
    """
    Creates a video from a sequence of frames.

    Parameters:
        frames (numpy.ndarray): Array of image frames (shape: N x H x W x C).
        output_video_file (str): Path to save the output video file.
        framerate (int, optional): Frames per second for the video. Default is 30.
    """
    frames = np.asarray(frames)

    # Normalize frames if values are in [0,1] range
    if frames.max() <= 1:
        frames = (frames * 255).astype(np.uint8)

    imageio.mimsave(output_video_file, frames, fps=framerate, quality=8)



class ProcessData(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    @torch.no_grad()
    def compute_rays(self, c2w, fxfycxcy, h=None, w=None, device="cuda"):
        """
        Args:
            c2w (torch.tensor): [b, v, 4, 4]
            fxfycxcy (torch.tensor): [b, v, 4]
            h (int): height of the image
            w (int): width of the image
        Returns:
            ray_o (torch.tensor): [b, v, 3, h, w]
            ray_d (torch.tensor): [b, v, 3, h, w]
        """

        b, v = c2w.size()[:2]
        c2w = c2w.reshape(b * v, 4, 4)

        fx, fy, cx, cy = fxfycxcy[:,:, 0], fxfycxcy[:,:,  1], fxfycxcy[:,:,  2], fxfycxcy[:,:,  3]
        h_orig = int(2 * cy.max().item())  # Original height (estimated from the intrinsic matrix)
        w_orig = int(2 * cx.max().item())  # Original width (estimated from the intrinsic matrix)
        if h is None or w is None:
            h, w = h_orig, w_orig

        # in case the ray/image map has different resolution than the original image
        if h_orig != h or w_orig != w:
            fx = fx * w / w_orig
            fy = fy * h / h_orig
            cx = cx * w / w_orig
            cy = cy * h / h_orig

        fxfycxcy = fxfycxcy.reshape(b * v, 4)
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        y, x = y.to(device), x.to(device)
        x = x[None, :, :].expand(b * v, -1, -1).reshape(b * v, -1)
        y = y[None, :, :].expand(b * v, -1, -1).reshape(b * v, -1)
        x = (x + 0.5 - fxfycxcy[:, 2:3]) / fxfycxcy[:, 0:1]
        y = (y + 0.5 - fxfycxcy[:, 3:4]) / fxfycxcy[:, 1:2]
        z = torch.ones_like(x)
        ray_d = torch.stack([x, y, z], dim=2)  # [b*v, h*w, 3]
        ray_d = torch.bmm(ray_d, c2w[:, :3, :3].transpose(1, 2))  # [b*v, h*w, 3]
        ray_d = ray_d / torch.norm(ray_d, dim=2, keepdim=True)  # [b*v, h*w, 3]
        ray_o = c2w[:, :3, 3][:, None, :].expand_as(ray_d)  # [b*v, h*w, 3]

        ray_o = rearrange(ray_o, "(b v) (h w) c -> b v c h w", b=b, v=v, h=h, w=w, c=3)
        ray_d = rearrange(ray_d, "(b v) (h w) c -> b v c h w", b=b, v=v, h=h, w=w, c=3)

        return ray_o, ray_d
    
    def fetch_views(self, data_batch, has_target_image=False, target_has_input=True):
        """
        Splits the input data batch into input and target sets.
        
        Args:
            data_batch (dict): Contains input tensors with the following keys:
                - 'image' (torch.Tensor): Shape [b, v, c, h, w], optional for some target views
                - 'fxfycxcy' (torch.Tensor): Shape [b, v, 4]
                - 'c2w' (torch.Tensor): Shape [b, v, 4, 4]
            target_has_input (bool): If True, target includes input views.

        Returns:
            tuple: (input_dict, target_dict), both as EasyDict objects.

        """
        # randomize input views if dynamic_input_view_num is True and not in inference mode
        if (self.config.training.get("dynamic_input_view_num", False) 
            and (not self.config.inference.get("if_inference", False))):
            self.config.training.num_input_views = np.random.randint(2, 5)
        

        input_dict, target_dict = {}, {}
        # index = [] save for future use if we want to select specific views

        num_target_views, num_views, bs, device = self.config.training.num_target_views, data_batch["c2w"].size(1), data_batch["image"].size(0), data_batch["image"].device
        assert num_target_views < num_views, f"We have {num_views} views, but we want to select {num_target_views} target views. This is more than the total number of views we have."
        
        # Decide the target view indices
        if self.config.training.dataset_name in ['data.dataset_scene.KubricDataset', 'data.dataset_scene.DL3DV']:
            # NOTE(Qingwen 205-05-22): always the last one:
            index = torch.tensor([
                [num_views - 1 - j for j in range(num_target_views)]
                for _ in range(bs)
            ], dtype=torch.long, device=device)
            # print("----- target views are the last ones, ", index)
        elif self.config.training.dataset_name in ['data.dataset_scene.RE10KDataset']:
            # Default one: Randomly sample target views across all views
            index = torch.tensor([
                random.sample(range(num_views), num_target_views)
                for _ in range(bs)
            ], dtype=torch.long, device=device) # [b, num_target_views]
            # print("----- target views are randomly sampled, ", index)
        else: 
            # NOTE: always the last one for other datasets or combined datasets
            index = torch.tensor([
                [num_views - 1 - j for j in range(num_target_views)]
                for _ in range(bs)
            ], dtype=torch.long, device=device)



        for key, value in data_batch.items():
            if key in ["scene_name", "avg_pose", "scene_scale"]:
                input_dict[key] = value
                target_dict[key] = value
                continue
            # FIXME(2025-09-30): here input always take the first num_input_views
            input_dict[key] = value[:, :self.config.training.num_input_views, ...]

            to_expand_dim = value.shape[2:] # [b, v, (value dim)] -> [value dim], e.g. [c, h, w] or [4] or [4, 4]
            expanded_index = index.view(index.shape[0], index.shape[1], *(1,) * len(to_expand_dim)).expand(-1, -1, *to_expand_dim)

            # Don't have target image supervision 
            if key == "image" and not has_target_image:                
                continue
            else:
                target_dict[key] = torch.gather(value, dim=1, index=expanded_index)
        
        height, width = data_batch["image"].shape[3], data_batch["image"].shape[4]
        input_dict["image_h_w"] = (height, width)
        target_dict["image_h_w"] = (height, width)

        input_dict, target_dict = edict(input_dict), edict(target_dict)

        return input_dict, target_dict


    
    @torch.no_grad()
    def forward(self, data_batch, has_target_image=True, target_has_input=True, compute_rays=True, ray_encoding="global"):
        """
        Preprocesses the input data batch and (optionally) computes ray_o and ray_d.

        Args:
            data_batch (dict): Contains input tensors with the following keys:
                - 'image' (torch.Tensor): Shape [b, v, c, h, w]
                - 'fxfycxcy' (torch.Tensor): Shape [b, v, 4]
                - 'c2w' (torch.Tensor): Shape [b, v, 4, 4]
            has_target_image (bool): If True, target views have image supervision.
            target_has_input (bool): If True, target views can be sampled from input views.
            compute_rays (bool): If True, compute ray_o and ray_d.
                
        Returns:
            Input and Target data_batch (dict): Contains processed tensors with the following keys:
                - 'image' (torch.Tensor): Shape [b, v, c, h, w]
                - 'fxfycxcy' (torch.Tensor): Shape [b, v, 4]
                - 'c2w' (torch.Tensor): Shape [b, v, 4, 4]
                - 'ray_o' (torch.Tensor): Shape [b, v, 3, h, w]
                - 'ray_d' (torch.Tensor): Shape [b, v, 3, h, w]
                - 'image_h_w' (tuple): (height, width)
        """
        input_dict, target_dict = self.fetch_views(data_batch, has_target_image=has_target_image, target_has_input=target_has_input)
        # print(f"--- [Debug] target_dict keys: {target_dict.keys()}, index: {target_dict.index}")
        if compute_rays:
            for dict in [input_dict, target_dict]:
                c2w = dict["c2w"]
                fxfycxcy = dict["fxfycxcy"]
                image_height, image_width = dict["image_h_w"]
                if ray_encoding == "global":
                    pass
                elif ray_encoding == "camray":
                    c2w = torch.eye(4, device=c2w.device, dtype=c2w.dtype).broadcast_to(c2w.shape)  # [b, v, 4, 4]
                ray_o, ray_d = self.compute_rays(c2w, fxfycxcy, image_height, image_width, device=data_batch["image"].device)
                dict["ray_o"], dict["ray_d"] = ray_o, ray_d

        return input_dict, target_dict

      
      