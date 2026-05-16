import torch

from omegaconf import OmegaConf
from gsplat import rasterization

from conerf.geometry.camera import Camera
from conerf.gaussian_fields.gaussian_splat_model import GaussianSplatModel


def render_gsplat(
    gaussian_splat_model: GaussianSplatModel,
    viewpoint_camera: Camera,
    pipeline_config: OmegaConf,
    bkgd_color: torch.Tensor,
    scaling_modifier: float = 1.0,
    anti_aliasing: bool = False,
    override_color: torch.Tensor = None,
    separate_sh: bool = False,
    sparse_grad: bool = False,
    packed: bool = False,
    # use_trained_exposure: bool = False,
    exposure: torch.Tensor = None,
    # Deblur related parameters.
    deblur_net: torch.nn.Module = None,
    lambda_scale: float = 0.01,
    lambda_position: float = 0.01,
    use_position_offset: bool = False,
    max_clamp: float = 1.1,
    device="cuda:0",
):
    means3D = gaussian_splat_model.get_xyz
    opacity = gaussian_splat_model.get_opacity
    K = viewpoint_camera.K
    scales = gaussian_splat_model.get_scaling * scaling_modifier
    rotations = gaussian_splat_model.get_quaternion

    if override_color is not None:
        colors = override_color  # [N,3]
        sh_degree = None
    else:
        colors = gaussian_splat_model.get_features  # [N,K,3]
        sh_degree = gaussian_splat_model.active_sh_degree

    viewmat = viewpoint_camera.world_to_camera.transpose(0, 1)

    if deblur_net is not None:
        _positions = means3D.detach()
        _scales = scales.detach()
        _rotations = rotations.detach()
        _viewdirs = viewpoint_camera.camera_center.repeat(means3D.shape[0], 1)

        delta_scales, delta_rotations, delta_positions = deblur_net(
            _positions, _scales, _rotations, _viewdirs)
        delta_scales = torch.clamp(
            lambda_scale * delta_scales + (1 - lambda_scale), min=1.0, max=max_clamp)
        delta_rotations = torch.clamp(
            lambda_scale * delta_rotations + (1 - lambda_scale), min=1.0, max=max_clamp)

        if not use_position_offset:  # Defocus blur.
            scales = scales * delta_scales
            rotations = rotations * delta_rotations
        else:  # Motion blur.
            delta_positions = lambda_position * delta_positions
            # Reshape to M 3D Gaussian sets.
            delta_positions = delta_positions.view(
                -1, 3, deblur_net.num_gaussian_sets - 1)
            delta_positions = torch.cat([
                delta_positions,
                torch.zeros((means3D.shape[0], 3, 1), dtype=means3D.dtype, device=means3D.device)
            ], dim=-1)
            delta_scales = delta_scales.view(-1,
                                             3, deblur_net.num_gaussian_sets)
            delta_rotations = delta_rotations.view(
                -1, 4, deblur_net.num_gaussian_sets)

            renders, radiis, depths = [], [], []
            screen_space_points_set, visibility_filters = [], []
            for i in range(deblur_net.num_gaussian_sets):
                positions = means3D + delta_positions[..., i]
                trans_scales = scales * delta_scales[..., i]
                trans_rotations = rotations * delta_rotations[..., i]

                render_colors, _, info = rasterization(
                    means=positions,
                    quats=trans_rotations,
                    scales=trans_scales,
                    opacities=opacity.squeeze(-1),
                    colors=colors,
                    viewmats=viewmat[None],
                    Ks=K,
                    backgrounds=bkgd_color[None],
                    width=int(viewpoint_camera.width),
                    height=int(viewpoint_camera.height),
                    rasterize_mode="classic" if not anti_aliasing else "antialiased",
                    sparse_grad=sparse_grad,
                    packed=packed,
                    sh_degree=sh_degree,
                    render_mode="RGB+ED",
                )

                rendered_image = render_colors[..., :3][0].permute(2, 0, 1)
                depth = render_colors[..., 3:][0].permute(2, 0, 1)
                radii = info["radii"].squeeze(0)  # [N,]
                try:
                    info["means2d"].retain_grad()  # [1,N,2]
                except:  # pylint: disable=W0702
                    pass
                
                renders.append(rendered_image)
                radiis.append(radii)
                depths.append(depth)
                screen_space_points_set.append(info["means2d"])
                visibility_filters.append(radii > 0)

            rendered_image = sum(renders) / len(renders)
            depth = sum(depths) / len(depths)
            return {
                "rendered_image": rendered_image,
                "screen_space_points": screen_space_points_set,
                "visibility_filter": visibility_filters,
                "radii": radiis,
                "scaling": scales,
                "depth": depths,
            }

    render_colors, _, info = rasterization(
        means=means3D,    # [N,3]
        quats=rotations,  # [N,4]
        scales=scales,    # [N,3]
        opacities=opacity.squeeze(-1),  # [N,]
        colors=colors,
        viewmats=viewmat[None],  # [1,4,4]
        Ks=K,  # [1,3,3]
        backgrounds=bkgd_color[None],
        width=int(viewpoint_camera.width),
        height=int(viewpoint_camera.height),
        rasterize_mode="classic" if not anti_aliasing else "antialiased",
        sparse_grad=sparse_grad,
        packed=packed,
        sh_degree=sh_degree,
        render_mode="RGB+ED",
    )

    rendered_image = render_colors[..., :3][0].permute(2, 0, 1)
    depth = render_colors[..., 3:][0].permute(2, 0, 1)
    radii = info["radii"].squeeze(0).max(dim=-1).values  # [N,]
    try:
        info["means2d"].retain_grad()  # [1,N,2]
    except:  # pylint: disable=W0702
        pass

    if exposure is not None:
        # Apply an affine transformation for each image to compensate for 
        # exposure changes.
        rendered_image = torch.matmul(
            rendered_image.permute(1, 2, 0), exposure[:3, :3]
        ).permute(2, 0, 1) + exposure[:3, 3, None, None]

    return {
        "rendered_image": rendered_image,  # [3,H,W]
        "screen_space_points": info["means2d"],
        "visibility_filter": radii > 0,
        "radii": radii,
        "scaling": scales,
        "depth": depth,
        "gaussian_ids": info["gaussian_ids"],
        "info": info,
    }