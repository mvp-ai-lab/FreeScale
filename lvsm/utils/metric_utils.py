import torch
from torch import Tensor
from jaxtyping import Float
from einops import reduce, rearrange
from skimage.metrics import structural_similarity
import functools
import os
from PIL import Image
from utils import data_utils
import numpy as np
from easydict import EasyDict as edict
import json
from rich import print

import warnings
# Suppress warnings for LPIPS loss loading
warnings.filterwarnings("ignore", category=UserWarning, message="The parameter 'pretrained' is deprecated since 0.13")
warnings.filterwarnings("ignore", category=UserWarning, message="Arguments other than a weight enum.*")

@torch.no_grad()
def compute_psnr(
    ground_truth: Float[Tensor, "b c h w"],
    predicted: Float[Tensor, "b c h w"],
    mask: Float[Tensor, "b 1 h w"] = None,
) -> Float[Tensor, "b"]:
    """
    Compute Peak Signal-to-Noise Ratio between ground truth and predicted images.
    
    Args:
        ground_truth: Images with shape [batch, channel, height, width], values in [0, 1]
        predicted: Images with shape [batch, channel, height, width], values in [0, 1]
        mask: Optional mask with shape [batch, channel, height, width], values > 0 for dynamic regions
        
    Returns:
        PSNR values for each image in the batch
    """
    ground_truth = torch.clamp(ground_truth, 0, 1)
    predicted = torch.clamp(predicted, 0, 1)

    if mask is not None:
        # Apply mask to both images - only compute MSE for dynamic regions
        mask_binary = (mask > 0).float()
        ground_truth_masked = ground_truth * mask_binary
        predicted_masked = predicted * mask_binary
        diff2 = (ground_truth_masked - predicted_masked) ** 2
        mse = reduce(diff2, "b c h w -> b", "sum") / (reduce(mask_binary, "b c h w -> b", "sum") + 1e-8)
    else:
        diff2 = (ground_truth - predicted) ** 2  # [b,c,h,w]
        mse = reduce(diff2, "b c h w -> b", "mean")
    return -10 * torch.log10(mse) 


@functools.lru_cache(maxsize=None)
def get_lpips_model(net_type="vgg", device="cuda"):
    # default lpips model
    from lpips import LPIPS
    # return LPIPS(net=net_type).to(device)

    # lpips_cst is a custom mask available version of LPIPS
    # NOTE(Qingwen): https://github.com/richzhang/PerceptualSimilarity/issues/111
    import sys, os
    BASE_DIR = os.path.abspath(os.path.dirname( __file__ ))
    sys.path.append(BASE_DIR)
    from lpips_cst import MaskLPIPS
    return LPIPS(net=net_type).to(device), MaskLPIPS(net=net_type).to(device)

@torch.no_grad()
def compute_lpips(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
    mask: Float[Tensor, "b c h w"] = None,
    normalize: bool = True,
) -> Float[Tensor, "batch"]:
    """
    Compute Learned Perceptual Image Patch Similarity between images.
    
    Args:
        ground_truth: Images with shape [batch, channel, height, width]
        predicted: Images with shape [batch, channel, height, width]
        The value range is [0, 1] when we have set the normalize flag to True.
        It will be [-1, 1] when the normalize flag is set to False.
    Returns:
        LPIPS values for each image in the batch (lower is better)
    """

    _lpips_fn, _lpips_fn_mask = get_lpips_model(device=predicted.device)
    batch_size = 10  # Process in batches to save memory
    # values = [
    #     _lpips_fn(
    #         ground_truth[i : i + batch_size],
    #         predicted[i : i + batch_size],
    #         normalize=normalize,
    #         mask=mask[i : i + batch_size] if mask is not None else None,
    #     )
    #     for i in range(0, ground_truth.shape[0], batch_size)
    # ]
    values = []
    for i in range(0, ground_truth.shape[0], batch_size):
        gt_batch = ground_truth[i : i + batch_size]
        pred_batch = predicted[i : i + batch_size]

        if mask is not None:
            mask_batch = mask[i : i + batch_size]
            lpips_value = _lpips_fn_mask(gt_batch, pred_batch, normalize=normalize, mask=mask_batch)
        else:
            lpips_value = _lpips_fn(gt_batch, pred_batch, normalize=normalize)

        values.append(lpips_value)

    lpips_ = torch.cat(values, dim=0).squeeze()
    if lpips_.dim() == 0:
        lpips_ = lpips_.unsqueeze(0)

    return lpips_



@torch.no_grad()
def compute_ssim(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
    mask: Float[Tensor, "b c h w"] = None,
) -> Float[Tensor, " batch"]:
    """
    Compute Structural Similarity Index between images.
    
    Args:
        ground_truth: Images with shape [batch, channel, height, width], values in [0, 1]
        predicted: Images with shape [batch, channel, height, width], values in [0, 1]
        
    Returns:
        SSIM values for each image in the batch (higher is better)
    """
    ssim_values= []
    
    for i_ in range(ground_truth.shape[0]):
        # Move to CPU and convert to numpy
        gt_np = ground_truth[i_].detach().cpu().numpy()
        pred_np = predicted[i_].detach().cpu().numpy()

        # Calculate SSIM
        _, ssim_map = structural_similarity(
            gt_np,
            pred_np,
            win_size=11,
            full=True, 
            gaussian_weights=True,
            channel_axis=0,
            data_range=1.0,
        )
        if mask is not None:
            mask_np = mask[i_].detach().cpu().numpy()
            ssim = np.mean(ssim_map[mask_np > 0])
        else:
            ssim = np.mean(ssim_map)

        ssim_values.append(ssim)
    
    # Convert back to tensor on the same device as input
    return torch.tensor(ssim_values, dtype=predicted.dtype, device=predicted.device)



@torch.no_grad()
def export_results(
    result: edict,
    data_path: str,
    out_dir: str, 
    compute_metrics: bool = False,
    custom_size: bool = False,
    target_cam: str = None
):
    """
    Save results including images and optional metrics and videos.
    
    Args:
        result: EasyDict containing input, target, and rendered images, and optionally video frames
        out_dir: Directory to save the evaluation results
        compute_metrics: Whether to compute and save metrics
    """
    os.makedirs(out_dir, exist_ok=True)
    
    input_data, target_data = result.input, result.target
    
    for batch_idx in range(input_data.image.size(0)):
        uid = input_data.index[batch_idx, 0, -1].item()
        scene_name = input_data.scene_name[batch_idx]
        sample_dir = os.path.join(out_dir, f"{uid:06d}")
        os.makedirs(sample_dir, exist_ok=True)
        
        # Get target view indices
        target_indices = target_data.index[batch_idx, :, 0].cpu().numpy()
        # print(f"Processing scene '{scene_name}' with target indices: {target_indices}")
        # Save images
        _save_images(result, batch_idx, sample_dir)
        
        # Compute and save metrics if requested
        if compute_metrics:
            _save_metrics(
                target_data.image[batch_idx],
                result.render[batch_idx],
                target_indices,
                sample_dir,
                scene_name,
                custom_size=custom_size,
                data_path=data_path,
                target_cam=target_cam
            )

        # Save video if available
        if hasattr(result, "video_rendering"):
            _save_video(result.video_rendering[batch_idx], sample_dir)

def visualize_intermediate_results(out_dir, result, max_batch_show=8, jpeg_quality=60):
    """
    jpeg_quality: PIL Image default is 75, higher is better, but for saving and check purpose we set 60 here.
    """
    os.makedirs(out_dir, exist_ok=True)

    input_data, target_data = result.input, result.target
    uid_based_filename = None  

    if result.render is not None:
        original_b = target_data.image.size(0)
        b = min(original_b, max_batch_show)

        target_image = target_data.image[:b]
        rendered_image = result.render[:b]
        target_index = target_data.index[:b]

        _, v, _, h, w = rendered_image.size()
        rendered_image = rendered_image.reshape(b * v, -1, h, w)
        target_image = target_image.reshape(b * v, -1, h, w)
        
        visualized_image = torch.cat((target_image, rendered_image), dim=3).detach().cpu()
        visualized_image = rearrange(visualized_image, "(b v) c h (m w) -> (b h) (v m w) c", v=v, m=2, b=b)
        visualized_image = (visualized_image.numpy() * 255.0).clip(0.0, 255.0).astype(np.uint8)
        
        uids = [target_index[i, 0, -1].item() for i in range(b)]
        uid_based_filename = f"{uids[0]:08}_{uids[-1]:08}"
        
        Image.fromarray(visualized_image).save(
            os.path.join(out_dir, f"supervision_{uid_based_filename}.jpg"),
            quality=jpeg_quality
        )
        with open(os.path.join(out_dir, f"uids.txt"), "w") as f:
            # uids_str = "_".join([f"{uid:08}" for uid in uids])
            uids_str = "Scene: \t\t\t Input&Output: \n"
            for b_ in range(b):
                uids_str += f"{target_data.scene_name[b_]}:"
                uids_str += f"{input_data.index[b_][...,0].tolist()}; {target_data.index[b_][...,0].tolist()}\n"
            f.write(uids_str)

    original_b_input = input_data.image.size(0)
    b_input = min(original_b_input, max_batch_show)

    input_images = input_data.image[:b_input]
    input_index = input_data.index[:b_input]

    input_uids = [input_index[i, 0, -1].item() for i in range(b_input)]
    input_uid_based_filename = f"{input_uids[0]:08}_{input_uids[-1]:08}"
    
    _, v, c, h, w = input_images.size()
    input_grid = input_images.reshape(b_input * v, c, h, w).detach().cpu()
    input_grid = rearrange(input_grid, "(b v) c h w -> (b h) (v w) c", v=v, b=b_input)
    input_grid = (input_grid.numpy() * 255.0).clip(0.0, 255.0).astype(np.uint8)
    
    Image.fromarray(input_grid).save(
        os.path.join(out_dir, f"input_{input_uid_based_filename}.jpg"),
        quality=jpeg_quality
    )
    
    return input_uid_based_filename, uid_based_filename

def visualize_ssl_results(
    out_dir,
    dynamic_input,
    dynamic_render,
    static_target,
    static_render,
    static_render2=None,
    max_batch_show=8,
    jpeg_quality=60,
):
    """
    为动态-静态模型的自监督学习 (SSL) 过程创建可视化结果。(已更新，支持 static_render2)

    此函数会生成两种图像：
    1.  `input_...jpg`: 动态模型的完整输入序列。
    2.  `supervision_...jpg`: 一个多栏对比图，从左到右依次为：
        - Panel 1: 左侧上下文帧 (动态模型的输入之一)
        - Panel 2..N: 动态模型的所有输出帧 (来自 dynamic_render)
        - Penultimate Panel: 静态模型的第一次输出 (来自 static_render)
        - Final Panel (可选): 静态模型的第二次输出 (来自 static_render2)

    参数:
        out_dir (str): 保存输出图像的目录。
        dynamic_input: 动态模型的输入数据对象。
        dynamic_render: 动态模型渲染的图像序列张量。
        static_target: 静态模型的目标数据对象 (用于获取UID等信息)。
        static_render: 静态模型的第一次渲染输出。
        static_render2 (Tensor, optional): 静态模型的第二次渲染输出。如果提供，将被添加到对比图的末尾。默认为 None。
        max_batch_show (int): 从一个批次中最多展示的样本数量。
        jpeg_quality (int): 保存JPEG图像时的压缩质量 (1-95)。
    """
    os.makedirs(out_dir, exist_ok=True)

    original_b_input = dynamic_input.image.size(0)
    b_input = min(original_b_input, max_batch_show)
    
    input_images_full = dynamic_input.image[:b_input]
    input_index = dynamic_input.index[:b_input]
    input_uids = [input_index[i, 0, -1].item() for i in range(b_input)]
    input_uid_based_filename = f"{input_uids[0]:08}_{input_uids[-1]:08}"
    
    _, v, c, h, w = input_images_full.size()
    input_grid = input_images_full.reshape(b_input * v, c, h, w).detach().cpu()
    input_grid = rearrange(input_grid, "(b v) c h w -> (b h) (v w) c", v=v, b=b_input)
    input_grid = (input_grid.numpy() * 255.0).clip(0.0, 255.0).astype(np.uint8)
    Image.fromarray(input_grid).save(
        os.path.join(out_dir, f"input_{input_uid_based_filename}.jpg"),
        quality=jpeg_quality
    )

    original_b = static_target.image.size(0)
    b = min(original_b, max_batch_show)
    
    len_dynamic_input = dynamic_input.image.shape[1]
    middle_frame_idx = len_dynamic_input // 2
    
    panels_to_concatenate = []

    context_frame_idx = max(0, middle_frame_idx - 1)
    left_context_image = dynamic_input.image[:b, context_frame_idx].unsqueeze(1)
    panels_to_concatenate.append(left_context_image)

    num_dynamic_renders = dynamic_render.shape[1]
    for i in range(num_dynamic_renders):
        panels_to_concatenate.append(dynamic_render[:b, i].unsqueeze(1))
    
    panels_to_concatenate.append(static_render[:b])
    
    if static_render2 is not None:
        num_static_renders2 = static_render2.shape[1]
        for i in range(num_static_renders2):
            # 确保每一帧都被正确地 unsqueeze 成 (b, 1, c, h, w)
            panels_to_concatenate.append(static_render2[:b, i].unsqueeze(1))
    
    visualized_image = torch.cat(panels_to_concatenate, dim=4).detach().cpu()
    
    total_panels = len(panels_to_concatenate)
    
    _, _, _, _, single_image_width = static_target.image.size()

    visualized_image = rearrange(
        visualized_image, "b v c h (m w) -> (b h) (v m w) c", v=1, m=total_panels, w=single_image_width
    )
    visualized_image = (visualized_image.numpy() * 255.0).clip(0.0, 255.0).astype(np.uint8)

    uids = [static_target.index[i, 0, -1].item() for i in range(b)]
    uid_based_filename = f"{uids[0]:08}_{uids[-1]:08}"

    Image.fromarray(visualized_image).save(
        os.path.join(out_dir, f"supervision_{uid_based_filename}.jpg"),
        quality=jpeg_quality
    )
    with open(os.path.join(out_dir, f"uids.txt"), "w") as f:
        uids_str = "_".join([f"{uid:08}" for uid in uids])
        f.write(uids_str)
        
    return input_uid_based_filename, uid_based_filename

def _save_images(result, batch_idx, out_dir):
    """Save visualization images."""
    # Save input image
    input_img = result.input.image[batch_idx]
    # if custom_size: # for a test only
    # input_img = upsample_and_crop(input_img, up_size=(512,512), crop_size=(342,512))
    input_img = rearrange(input_img, "v c h w -> h (v w) c")
    input_img = (input_img.cpu().numpy() * 255.0).clip(0.0, 255.0).astype(np.uint8)
    Image.fromarray(input_img).save(os.path.join(out_dir, "input.png"))

    # Save GT vs prediction side-by-side
    comparison = torch.cat((result.target.image[batch_idx], result.render[batch_idx]), dim=3).detach().cpu()
    comparison = rearrange(comparison, "v c h w -> h (v w) c")
    comparison = (comparison.numpy() * 255.0).clip(0.0, 255.0).astype(np.uint8)
    Image.fromarray(comparison).save(os.path.join(out_dir, "gt_vs_pred.png"))

import torch
import torch.nn.functional as F

def upsample_and_crop(
    img: torch.FloatTensor,          # shape [B, C, 256, 256]
    up_size: tuple = (512, 512),     # (H, W) to upsample to
    crop_size: tuple = (342, 512),   # (H_crop, W_crop) to extract
    mode: str = 'area',
    center_crop: bool = True
) -> torch.FloatTensor:
    """
    Upsample a batch of images to up_size, then crop out a region of size crop_size.

    Args:
        img:         [B, C, H0, W0] tensor (e.g. 256x256)
        up_size:     target (H, W) after interpolation
        crop_size:   region to crop (H_crop, W_crop). Must be less-or-equal to up_size.
        mode:        interpolation mode for upsampling
        align_corners: passed to F.interpolate
        center_crop: if True, crop centered; else crop from top-left
    Returns:
        cropped: [B, C, H_crop, W_crop]
    """
    # 1) Upsample
    up = F.interpolate(img, size=up_size, mode=mode)

    # 2) Compute crop coordinates
    H_up, W_up = up_size
    Hc, Wc = crop_size
    if center_crop:
        top  = (H_up - Hc) // 2
        left = (W_up - Wc) // 2
    else:
        top, left = 0, 0
    bottom = top + Hc
    right  = left + Wc

    # 3) Slice out the crop
    return up[:, :, top:bottom, left:right]

import cv2
def _save_metrics(target, prediction, view_indices, out_dir, scene_name, custom_size=True, data_path=None, target_cam=None):
    target = target.to(torch.float32) # [v, 3, h, w]
    prediction = prediction.to(torch.float32)
    if custom_size:
        target = upsample_and_crop(target, up_size=(512,512), crop_size=(342,512))
        prediction = upsample_and_crop(prediction, up_size=(512,512), crop_size=(342,512))
        
    psnr_values = compute_psnr(target, prediction)
    lpips_values = compute_lpips(target, prediction)
    ssim_values = compute_ssim(target, prediction)
    # print(f"PSNR: {psnr_values}, LPIPS: {lpips_values}, SSIM: {ssim_values}")
    # dynamic only part:
    if custom_size and data_path is not None and target_cam is not None and (isinstance(target_cam, list) and len(target_cam) == 1):
        target_cam = target_cam[0] if isinstance(target_cam, list) else target_cam
        # FIXME(Qingwen): hardcoded index here! Please change it to align with target index later.
        path_dynamic_mask = f"{data_path}/{scene_name}/{target_cam}/segmentation_00020.png"
        dynamic_mask = cv2.imread(path_dynamic_mask)
        dynamic_mask = torch.from_numpy(dynamic_mask).permute(2, 0, 1).unsqueeze(0).to(torch.float32).to(prediction.device)
        dynamic_mask = upsample_and_crop(dynamic_mask, up_size=(512,512), crop_size=(342,512)) > 0
        dynamic_psnr = compute_psnr(target, prediction, mask=dynamic_mask)
        dynamic_ssim = compute_ssim(target, prediction, mask=dynamic_mask)
        dynamic_lpips = compute_lpips(target, prediction, mask=dynamic_mask)
    else:
        custom_size = False # hardcode for afterward flag
    metrics = {
        "summary": {
            "scene_name": scene_name,
            "psnr": float(psnr_values.mean()),
            "lpips": float(lpips_values.mean()),
            "ssim": float(ssim_values.mean()),
            "dynamic_psnr": float(dynamic_psnr.mean()) if custom_size else -1.0,
            "dynamic_lpips": float(dynamic_lpips.mean()) if custom_size else -1.0,
            "dynamic_ssim": float(dynamic_ssim.mean()) if custom_size else -1.0,
        },
        "per_view": []
    }
    # print(metrics)
    for i, view_idx in enumerate(view_indices):
        metrics["per_view"].append({
            "view": int(view_idx), "psnr": float(psnr_values[i]), "lpips": float(lpips_values[i]), "ssim": float(ssim_values[i]), \
            "dynamic_psnr": float(dynamic_psnr[i]) if custom_size else -1.0, \
            "dynamic_lpips": float(dynamic_lpips[i]) if custom_size else -1.0, \
            "dynamic_ssim": float(dynamic_ssim[i]) if custom_size else -1.0,
        })
    
    # Save metrics to a single JSON file
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


def _save_video(frames, out_dir):
    """
    Save video from rendered frames.
    Input frames should be in [v, c, h, w] format.
    """
    frames = np.ascontiguousarray(np.array(frames.to(torch.float32)))
    frames = rearrange(frames, "v c h w -> v h w c")
    data_utils.create_video_from_frames(
        frames, 
        f"{out_dir}/rendered_video.mp4", 
        framerate=30
    )


def summarize_evaluation(evaluation_folder):
    # Find and sort all valid subfolders
    subfolders = sorted(
        [
            os.path.join(evaluation_folder, dirname)
            for dirname in os.listdir(evaluation_folder)
            if os.path.isdir(os.path.join(evaluation_folder, dirname))
        ],
        key=lambda x: int(os.path.basename(x)) if os.path.basename(x).isdigit() else os.path.basename(x)
    )

    metrics = {}
    valid_subfolders = []
    
    for subfolder in subfolders:
        json_path = os.path.join(subfolder, "metrics.json")
        if not os.path.exists(json_path):
            print(f"!!! Metrics file not found in {subfolder}, skipping...")
            continue
            
        valid_subfolders.append(subfolder)
        
        with open(json_path, "r") as f:
            try:
                data = json.load(f)
                # Extract summary metrics
                for metric_name, metric_value in data["summary"].items():
                    if metric_name == "scene_name":
                        continue
                    metrics.setdefault(metric_name, []).append(metric_value)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error reading metrics from {json_path}: {e}")

    if not valid_subfolders:
        print(f"No valid metrics files found in {evaluation_folder}")
        return

    csv_file = os.path.join(evaluation_folder, "summary.csv")
    with open(csv_file, "w") as f:
        header = ["Index"] + list(metrics.keys())
        f.write(",".join(header) + "\n")
        
        for i, subfolder in enumerate(valid_subfolders):
            basename = os.path.basename(subfolder)
            values = [str(metric_values[i]) for metric_values in metrics.values()]
            f.write(f"{basename},{','.join(values)}\n")
        
        f.write("\n")
        
        averages = [str(sum(values) / len(values)) for values in metrics.values()]
        f.write(f"average,{','.join(averages)}\n")
    
    print(f"Summary written to {csv_file}")
    print(f"----\nAverage Score: \nPSNR: {averages[0]}, LPIPS: {averages[1]}, SSIM: {averages[2]}, \nDynamic PSNR: {averages[3]}, Dynamic LPIPS: {averages[4]}, Dynamic SSIM: {averages[5]}\n----")
    print(f"{averages[0]},{averages[2]},{averages[1]},{averages[3]},{averages[5]},{averages[4]}")
    # export average metrics to a text file
    with open(os.path.join(evaluation_folder, "average_metrics.txt"), "w") as f:
        f.write(f"Average: {','.join(averages)}\n")
    