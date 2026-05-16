# pylint: disable=[E1101,W0621]

import copy
import os
import time
import json
from typing import List, Literal
import tqdm

import torch
import torch.nn.functional as F
import imageio.v2 as imageio
from omegaconf import OmegaConf
from PIL import Image

from conerf.render.gaussian_render import render_gsplat  
from conerf.base.checkpoint_manager import CheckPointManager
from conerf.evaluators.evaluator import (
    Evaluator, compute_psnr, compute_lpips, compute_ssim, color_correct
)
from conerf.geometry.camera import Camera
from conerf.gaussian_fields.app_embed import AppearanceOptModule
from conerf.gaussian_fields.gaussian_splat_model import GaussianSplatModel
from conerf.utils.utils import save_images, get_subdirs, colorize
from scripts.preprocess.utils import list_images

from fused_ssim import fused_ssim


@torch.enable_grad()
def optimize_embedding(
    config: OmegaConf,
    model: GaussianSplatModel,
    app_module: torch.nn.Module,
    cameras: List[Camera],
    color_bkgd: torch.Tensor,
    device,
):
    from conerf.render.gaussian_render import render_gsplat  # pylint: disable=C0415

    model.eval()

    appearance_embedding_optim_iters = 128
    app_module.embeds = torch.nn.Embedding(
        len(cameras), app_module.embed_dim).to(device)
    optimizer = torch.optim.Adam(
        app_module.embeds.parameters(),
        lr=config.optimizer.lr_app.app_module * 10.0, weight_decay=1e-6
    )
    # Freeze the MLP layer as we only optimize the per-image appearance embedding.
    for param in app_module.color_head.parameters():
        param.requires_grad = False

    pbar = tqdm.trange(
        len(cameras), desc="Test-time optimization for appearance embedding...")
    for i, camera in enumerate(cameras):
        camera = camera.copy_to_device(device)
        for _ in range(appearance_embedding_optim_iters):
            optimizer.zero_grad()

            dirs = model.get_xyz - camera.camera_center.repeat(
                model.get_features.shape[0], 1)
            precompute_colors = app_module(
                features=model.get_features,
                embed_ids=torch.tensor([i], device=device),
                dirs=dirs[None],
                sh_degree=model.max_sh_degree,
            )
            precompute_colors = precompute_colors + model.get_colors
            precompute_colors = torch.sigmoid(precompute_colors)

            colors = render_gsplat(
                gaussian_splat_model=model,
                viewpoint_camera=camera,
                pipeline_config=config.pipeline,
                bkgd_color=color_bkgd,
                anti_aliasing=config.texture.anti_aliasing,
                separate_sh=False,
                exposure=None,
                override_color=precompute_colors,
                device=device,
            )["rendered_image"]
            pixels = camera.image.permute(2, 0, 1).to(device)

            lambda_dssim = config.loss.lambda_dssim
            loss_ssim = fused_ssim(colors.unsqueeze(0), pixels.unsqueeze(0))
            loss_rgb_l1 = F.l1_loss(colors, pixels)
            loss = (1.0 - lambda_dssim) * loss_rgb_l1 + \
                lambda_dssim * (1.0 - loss_ssim)
            loss.backward()

            optimizer.step()
        # End of optimization.
        pbar.update(1)


class GaussianSplatEvaluator(Evaluator):
    """Class for evaluating NeRF models."""

    def __init__(
        self,
        config: OmegaConf,
        load_train_data: bool = False,
        trainset=None,
        load_val_data: bool = True,
        valset=None,
        load_test_data: bool = False,
        testset=None,
        models: List = None,
        meta_data: List = None,
        verbose: bool = False,
        device: str = "cuda",
    ) -> None:
        self.app_module = None
        if config.appearance.use_app_embed:
            feature_dim = config.appearance.app_feat_dim
            embed_dim = config.appearance.app_embed_dim
            self.app_module = AppearanceOptModule(
                0,
                feature_dim,
                embed_dim,
                config.texture.max_sh_degree,
            ).to(device)

        super().__init__(
            config,
            load_train_data,
            trainset,
            load_val_data,
            valset,
            load_test_data,
            testset,
            models,
            meta_data,
            verbose,
            device,
        )

        self.color_bkgd = torch.tensor(
            [0, 0, 0], dtype=torch.float32, device=self.device)

        if self.app_module is not None and self.app_module.embeds.weight.shape[0] == 0:
            embed_dim = self.config.appearance.app_embed_dim
            self.app_module.embeds = torch.nn.Embedding(
                len(self.val_dataset.cameras), embed_dim
            ).to(self.device)

    def _prepare_model_init_params(self, model_type: Literal["global", "local"] = "local"):
        pass

    def _build_networks(self, *args, **kwargs):  # pylint: disable=[W0613]
        model = GaussianSplatModel(
            max_sh_degree=self.config.texture.max_sh_degree,
            percent_dense=self.config.geometry.percent_dense,
            app_feat_dim=self.config.appearance.get("app_feat_dim", None) if
            self.config.appearance.use_app_embed else None,
        )
        return model

    def setup_metadata(self):
        """Set up meta data that are required to initialize/evaluate a model."""
        # meta data for construction models.
        meta_data = {
            "active_sh_degree": None,
            "xyz": None,
            "features_dc": None,
            # "features_rest": None,
            "scaling": None,
            "quaternion": None,
            "opacity": None,
            "max_radii2D": None,
            "xyz_gradient_accum": None,
            "denom": None,
            "spatial_lr_scale": None,
            "num_train_images": None,
        }
        if self.config.appearance.use_app_embed:
            meta_data["color"] = None
        else:
            meta_data["features_rest"] = None

        if self.config.dataset.multi_blocks:
            meta_data["block_id"] = None

        return meta_data

    def load_model(self):
        self.meta_data, self.models = [], []  # pylint: disable=W0201
        ckpt_manager = CheckPointManager(verbose=False)

        input_model_dir = os.path.join(
            self.config.dataset.output_dir, self.config.expname)

        assert os.path.exists(input_model_dir), \
            f"input model directory does not exist: {input_model_dir}"
        if self.config.dataset.multi_blocks:
            model_dirs = get_subdirs(input_model_dir, "block_")
        else:
            model_dirs = [input_model_dir]

        pbar = tqdm.trange(len(model_dirs), desc="Loading Models", leave=True)
        for model_dir in model_dirs:
            local_config = copy.deepcopy(self.config.trainer)
            local_config.ckpt_path = os.path.join(model_dir, 'model.pth')
            assert os.path.exists(local_config.ckpt_path), \
                f"checkpoint does not exist: {local_config.ckpt_path}"

            with open(
                os.path.join(model_dir, "meta_non_splats.json"), "r", encoding="utf-8"
            ) as f:
                non_splat_meta_data = json.load(f)

            model = self._build_networks()
            model.active_sh_degree = non_splat_meta_data["active_sh_degree"]

            print(f'Loading model from {model_dir}')
            if self.config.compression.get("enabled", False):
                model.decompress(model_dir)
            else:
                model.load_ply(os.path.join(model_dir, 'final_splat.ply'))

            iteration = non_splat_meta_data["iteration"]
            self.meta_data.append(non_splat_meta_data)
            self.model_iterations.append(iteration)

            if self.config.appearance.use_app_embed:
                embed_dim = self.config.appearance.app_embed_dim
                # Reset appearance embedding to have length equal to training views
                # for loading checkpoints, though the we don't need appearance embedding
                # from training views during evaluation.
                self.app_module.embeds = torch.nn.Embedding(
                    self.config.appearance.input_dim, embed_dim
                ).to(self.device)
                ckpt_manager.load(local_config, models={
                                  "app_module": self.app_module})
                # Reset appearance embedding to have length equal to evaluation views.
                # We will optimize the embeddings during evaluation.
                self.app_module.embeds = torch.nn.Embedding(
                    len(self.val_dataset.cameras), embed_dim
                ).to(self.device)

            self.models.append(model)

            pbar.update(1)

    def eval(
        self,
        iteration: int = None,
        split: Literal["val", "test"] = "val",
    ) -> dict:
        """
        Main logic for evaluation.
        """
        metrics = dict()
        if split == "val":
            assert self.val_dataset is not None
            dataset = self.val_dataset
        elif split == "test":
            assert self.test_dataset is not None
            dataset = self.test_dataset
        else:
            if self.verbose:
                print(f'[WARNING] {split} set does not exist!')
            return

        eval_dir = os.path.join(self.eval_dir, split)
        os.makedirs(eval_dir, exist_ok=True)

        num_blocks = len(self.models)

        data_dir = os.path.join(self.config.dataset.root_dir, self.config.dataset.scene)
        if self.config.dataset.root_dir.lower().find("nerfbusters") >= 0:
            vis_map_dir = os.path.join(data_dir, "visibility_median_count")
            vis_map_paths = list_images(vis_map_dir)
        else:
            vis_map_paths = None

        meta_data = self.meta_data[0]
        meta_data["split"] = split
        for k, model in enumerate(self.models):
            if model.get_xyz.device == torch.device('cpu'):
                model = model.to("cuda")
            else:
                model.eval()

            # meta_data = self.meta_data[k]
            iteration = self.model_iterations[k] if iteration is None else iteration

            val_dir = eval_dir
            if self.config.dataset.multi_blocks and len(self.models) > 1:
                val_dir = os.path.join(eval_dir, f"block_{k}")
            os.makedirs(val_dir, exist_ok=True)

            if self.verbose:
                print(f'Results are saving to: {val_dir}')

            image_dir = os.path.join(val_dir, "images")
            # if iteration is not None:
            #     image_dir = os.path.join(image_dir, f'iter_{iteration}')
            os.makedirs(image_dir, exist_ok=True)

            cameras = dataset.cameras
            if self.app_module is not None and split == "val":
                optimize_embedding(
                    self.config, model, self.app_module, cameras, self.color_bkgd, self.device
                )

            pbar = tqdm.trange(
                len(cameras), desc=f"Validating {self.config.expname}", leave=False
            )
            psnrs, ssims, lpips, render_times, render_mems = {}, {}, {}, {}, {}

            for i in range(len(cameras)):  # pylint: disable=C0200
                camera = cameras[i]
                camera = camera.copy_to_device(self.device)

                if vis_map_paths is not None:
                    pseudo_gt_visibility = torch.from_numpy(
                        imageio.imread(vis_map_paths[i])
                    ).float().to(self.device)
                    visibility_mask = (pseudo_gt_visibility[..., 0] >= 1).float()
                    meta_data["mask"] = visibility_mask[None, ...].repeat(3, 1, 1)
                else:
                    meta_data["mask"] = None

                psnrs[i], ssims[i], lpips[i], render_times[i], render_mems[i] = self._eval(
                    camera, model, meta_data, image_dir, i
                )

                pbar.update(1)

            avg_psnr = sum(psnrs.values()) / len(psnrs)
            avg_ssim = sum(ssims.values()) / len(ssims)
            avg_lpips = sum(lpips.values()) / len(lpips)
            avg_time = sum(render_times.values()) / len(render_times)
            avg_mem = sum(render_mems.values()) / len(render_mems)

            metric_key = k if k < num_blocks else "global"
            metrics[metric_key] = {
                'iteration': iteration,
                'all_psnr': psnrs,
                'all_ssim': ssims,
                'all_lpips': lpips,
                'all_times': render_times,
                'all_mems': render_mems,
                'psnr': avg_psnr,
                'ssim': avg_ssim,
                'lpips': avg_lpips,
                'time': avg_time,
                'memory': avg_mem,
                "points": model.get_xyz.shape[0],
            }

            if split == "test":
                video_name = os.path.join(eval_dir, 'render.mp4')
                rendered_image_dir = os.path.join(image_dir, "rgb_test")
                os.system(
                    f"ffmpeg -framerate 10 -i {rendered_image_dir}/%3d.png " +
                    "-vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' " +
                    f"-c:v libx264 -pix_fmt yuv420p {video_name}")

        metric_file = os.path.join(eval_dir, 'metrics.json')
        json_obj = json.dumps(metrics, indent=4)
        if self.verbose:
            print(f'Saving metrics to {metric_file}')
        with open(metric_file, 'a', encoding='utf-8') as json_file:
            json_file.write(json_obj)

        return metrics

    @torch.no_grad()
    def _eval(self, data, model, meta_data, eval_dir, image_index):  # pylint: disable=W0613
        pixels = data.image  # [height, width, RGB]

        # rendering
        torch.cuda.reset_peak_memory_stats()
        time_start = time.time()

        precompute_colors = None
        if self.app_module is not None:
            dirs = model.get_xyz - data.camera_center.repeat(
                model.get_features.shape[0], 1)
            embed_ids = torch.tensor([image_index], device=self.device) \
                if meta_data["split"] == "val" else None
            precompute_colors = self.app_module(
                features=model.get_features,
                embed_ids=embed_ids,
                dirs=dirs[None],
                sh_degree=model.max_sh_degree,
                embed_value=0,
            )
            precompute_colors = precompute_colors + model.get_colors
            precompute_colors = torch.sigmoid(precompute_colors)

        render_results = render_gsplat(  # pylint: disable=E0606
            gaussian_splat_model=model,
            viewpoint_camera=data,
            pipeline_config=self.config.pipeline,
            bkgd_color=self.color_bkgd,
            anti_aliasing=self.config.texture.anti_aliasing,
            separate_sh=False,  # True,
            override_color=precompute_colors,
        )

        render_time = time.time() - time_start
        render_max_mem = torch.cuda.max_memory_allocated() / (1024.0 ** 2)

        colors, screen_space_points, visibility_filter, radii = (  # pylint: disable=W0612
            render_results["rendered_image"],
            render_results["screen_space_points"],
            render_results["visibility_filter"],
            render_results["radii"],
        )
        colors, depth = render_results["rendered_image"], render_results["depth"]
        mask = meta_data["mask"]
        if mask is not None:
            pixels = pixels * mask.permute(1, 2, 0)
            colors = colors * mask
            depth = depth * mask[0:1, :, :,]

        pixels, colors = pixels.cpu(), colors.cpu()
        colors = torch.clamp(colors, 0, 1)
        depth = colorize(depth.cpu().squeeze(0), cmap_name="jet")

        if meta_data["split"] == "val" and self.config.evaluator.get("correct_color", False):
            colors_cc = color_correct(colors.permute(
                1, 2, 0).numpy(), pixels.numpy())
            colors_cc = torch.from_numpy(colors_cc).permute(2, 0, 1)
        else:
            colors_cc = colors

        image_dict = {}
        image_dict["rgb_gt"] = pixels
        image_dict["rgb_test"] = colors_cc.permute(1, 2, 0)
        image_dict["depth"] = depth

        save_images(save_dir=eval_dir,
                    image_dict=image_dict, index=image_index)

        if meta_data["split"] == "val":
            pixels = pixels[None, ...].to(self.device).permute(0, 3, 1, 2)
            colors_cc = colors_cc[None, ...].to(self.device)
            psnr = compute_psnr(pixels, colors_cc).item()
            ssim = compute_ssim(pixels, colors_cc)
            lpips = compute_lpips(self.lpips_loss, pixels, colors_cc)
        else:
            psnr, ssim, lpips = 0, 0, 0

        return psnr, ssim, lpips, render_time, render_max_mem

    def _export_mesh(self, model, iteration, mesh_dir):
        pass

    def _export_mesh(self, model, iteration, mesh_dir):
        pass

    def denoise_eval(
        self,
        iteration: int = None,
        split: Literal["val", "test"] = "val",
    ) -> dict:
        """
        Main logic for evaluation.
        """
        metrics = dict()
        if split == "val":
            assert self.val_dataset is not None
            dataset = self.val_dataset
        elif split == "test":
            assert self.test_dataset is not None
            dataset = self.test_dataset
        else:
            if self.verbose:
                print(f'[WARNING] {split} set does not exist!')
            return

        eval_dir = os.path.join(self.eval_dir, split)
        os.makedirs(eval_dir, exist_ok=True)

        num_blocks = len(self.models)

        meta_data = self.meta_data[0]
        meta_data["split"] = split
        for k, model in enumerate(self.models):
            if model.get_xyz.device == torch.device('cpu'):
                model = model.to("cuda")
            else:
                model.eval()

            # meta_data = self.meta_data[k]
            iteration = self.model_iterations[k] if iteration is None else iteration

            val_dir = eval_dir
            if self.config.dataset.multi_blocks and len(self.models) > 1:
                val_dir = os.path.join(eval_dir, f"block_{k}")
            os.makedirs(val_dir, exist_ok=True)

            if self.verbose:
                print(f'Results are saving to: {val_dir}')

            image_dir = os.path.join(val_dir, "images")
            # if iteration is not None:
            #     image_dir = os.path.join(image_dir, f'iter_{iteration}')
            os.makedirs(image_dir, exist_ok=True)

            cameras = dataset.cameras
            if self.app_module is not None and split == "val":
                optimize_embedding(
                    self.config, model, self.app_module, cameras, self.color_bkgd, self.device
                )

            pbar = tqdm.trange(
                len(cameras), desc=f"Validating {self.config.expname}", leave=False
            )
            psnrs, ssims, lpips, render_times, render_mems = {}, {}, {}, {}, {}

            for i in range(len(cameras)):  # pylint: disable=C0200
                camera = cameras[i]
                camera = camera.copy_to_device(self.device)

                psnrs[i], ssims[i], lpips[i], render_times[i], render_mems[i] = self._eval(
                    camera, model, meta_data, image_dir, i
                )

                pbar.update(1)

            avg_psnr = sum(psnrs.values()) / len(psnrs)
            avg_ssim = sum(ssims.values()) / len(ssims)
            avg_lpips = sum(lpips.values()) / len(lpips)
            avg_time = sum(render_times.values()) / len(render_times)
            avg_mem = sum(render_mems.values()) / len(render_mems)

            metric_key = k if k < num_blocks else "global"
            metrics[metric_key] = {
                'iteration': iteration,
                'all_psnr': psnrs,
                'all_ssim': ssims,
                'all_lpips': lpips,
                'all_times': render_times,
                'all_mems': render_mems,
                'psnr': avg_psnr,
                'ssim': avg_ssim,
                'lpips': avg_lpips,
                'time': avg_time,
                'memory': avg_mem,
                "points": model.get_xyz.shape[0],
            }

            if split == "test":
                video_name = os.path.join(eval_dir, 'render.mp4')
                rendered_image_dir = os.path.join(image_dir, "rgb_test")
                os.system(
                    f"ffmpeg -framerate 10 -i {rendered_image_dir}/%3d.png " +
                    "-vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' " +
                    f"-c:v libx264 -pix_fmt yuv420p {video_name}")

        metric_file = os.path.join(eval_dir, 'metrics_denoise.json')
        json_obj = json.dumps(metrics, indent=4)
        if self.verbose:
            print(f'Saving metrics to {metric_file}')
        with open(metric_file, 'a', encoding='utf-8') as json_file:
            json_file.write(json_obj)

        return metrics

    @torch.no_grad()
    def run_difix(self, view_graph=None):
        freeviews_dir = os.path.join(self.eval_dir, "freeviews")
        if os.path.exists(f"{freeviews_dir}"):
            os.system(f"rm -rf {freeviews_dir}")
        os.makedirs(f"{freeviews_dir}/difix/", exist_ok=True)
        os.makedirs(f"{freeviews_dir}/ref/", exist_ok=True)
        if self.app_module is not None:
            for app_i in range(2):
                os.makedirs(f"{freeviews_dir}/difix_{app_i+1}/", exist_ok=True)

        orig_candidates = torch.load(os.path.join(self.eval_dir, 'cameras.pt'), weights_only=False)
        image_poses = [cam.cam_to_world.cpu().numpy() for cam in orig_candidates] 

        train_poses = torch.stack([cam.cam_to_world for cam in self.train_cameras])
        train_poses = train_poses.cpu().numpy()
        ref_trainid = self.interpolator.find_nearest_assignments(train_poses, image_poses)
        
        for ci, cam in enumerate(orig_candidates):
            i = int(cam.image_index.split("_")[1])
            candidate = copy.deepcopy(cam)
            image_path = os.path.join(self.eval_dir, cam.image_path)
            assert os.path.exists(image_path), f"{image_path}"
            image = Image.open(image_path).convert("RGB")

            ref_image = self.get_ref_fromVG(candidate.image_index, view_graph)
            if ref_image is None:
                ref_image = Image.open(self.train_cameras[ref_trainid[ci]].image_path).convert("RGB")
                ref_image = ref_image.resize(image.size, Image.BILINEAR)

            output_image = self.difix(prompt="remove degradation", image=image, ref_image=ref_image, 
                    num_images_per_prompt=1, num_inference_steps=1, timesteps=[199], guidance_scale=0.0).images[0]
            output_image = output_image.resize(image.size, Image.LANCZOS)
            output_image.save(f"{freeviews_dir}/difix/{i:03d}.png")


    
    @torch.no_grad()
    def get_ref_fromVG(self, ind, view_graph):
        neigbours = view_graph[ind]
        if len(neigbours) == 0:
            return None
        
        success = False
        num_try = 0
        while not success and num_try < 2:
            for n in neigbours:
                nid = n[0]
                if isinstance(nid, int) or nid.find("fv") < 0:
                    success = True
                    break
            if num_try >= len(neigbours):
                break
            neigbours = view_graph[str(neigbours[num_try][0])]
            num_try += 1

        if success:
            ref_path = self.train_cameras[int(nid)].image_path
            ref_image = Image.open(ref_path).convert("RGB")
        else:
            ref_image = None
        return ref_image
