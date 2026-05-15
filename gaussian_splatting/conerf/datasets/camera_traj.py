import torch
import numpy as np
import torch.nn.functional as F
import random
import roma
import scipy


def calculate_fov(intrinsics):
    def focal_length_to_fov(focal_length, pixels):
        return 2 * np.arctan(pixels / (2 * focal_length))

    f = np.array([intrinsics[0, 0], intrinsics[1, 1]])
    c = np.array([intrinsics[0, 2], intrinsics[1, 2]])
    size = (c * 2).astype(int)

    fov = focal_length_to_fov(f, size)
    return fov


def get_intrinsics(fov, intrinsics):
    if not isinstance(fov, torch.Tensor):
        fov = torch.as_tensor(fov, dtype=torch.float32,
                              device=intrinsics.device)
    if not isinstance(intrinsics, torch.Tensor):
        intrinsics = torch.as_tensor(
            intrinsics, dtype=torch.float32, device=intrinsics.device)
    c = intrinsics[:2, 2]
    size = (c * 2)
    f = size / (2 * torch.tan(fov / 2.0))

    new_intrinsics = intrinsics[None, :, :].repeat(
        fov.shape[0], 1, 1)  # (N, 3, 3)
    new_intrinsics[:, 0, 0] = f[:, 0]  # fx
    new_intrinsics[:, 1, 1] = f[:, 1]  # fy
    return new_intrinsics


def rt_to_mat4(
    R: torch.Tensor, t: torch.Tensor, s: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Args:
        R (torch.Tensor): (..., 3, 3).
        t (torch.Tensor): (..., 3).
        s (torch.Tensor): (...,).

    Returns:
        torch.Tensor: (..., 4, 4)
    """
    mat34 = torch.cat([R, t[..., None]], dim=-1)
    if s is None:
        bottom = (
            mat34.new_tensor([[0.0, 0.0, 0.0, 1.0]])
            .reshape((1,) * (mat34.dim() - 2) + (1, 4))
            .expand(mat34.shape[:-2] + (1, 4))
        )
    else:
        bottom = F.pad(1.0 / s[..., None, None], (3, 0), value=0.0)
    mat4 = torch.cat([mat34, bottom], dim=-2)
    return mat4


def get_lookat_w2cs(
    positions: torch.Tensor,
    lookat: torch.Tensor,
    up: torch.Tensor,
    face_off: bool = False,
):
    """
    Args:
        positions: (B, N, 3) tensor of camera positions
        lookat: (B, 3,) tensor of lookat point
        up: (B, 3,) or (B, N, 3) tensor of up vector

    Returns:
        w2cs: (B, N, 4, 4) tensor of world to camera rotation matrices
    """
    if len(lookat.shape) == 2:
        lookat = lookat.unsqueeze(1)
    forward_vectors = F.normalize(lookat - positions, dim=-1)
    if face_off:
        forward_vectors = -forward_vectors
    if up.dim() == 2:
        up = up.unsqueeze(1)
    right_vectors = F.normalize(torch.cross(
        forward_vectors, up, dim=-1), dim=-1)
    down_vectors = F.normalize(
        torch.cross(forward_vectors, right_vectors, dim=-1), dim=-1
    )
    Rs = torch.stack([right_vectors, down_vectors, forward_vectors], dim=-1)
    w2cs = torch.linalg.inv(rt_to_mat4(Rs, positions))
    return w2cs


def create_preset_poses(
    option, cameras, look_at, up_direction, K, n_steps=120, zoom_factor=1.0
):
    if len(cameras.shape) == 2:
        cameras = cameras[None, ...]
    start_w2c = torch.linalg.inv(cameras)

    if option == "orbit":
        camtoworld = torch.linalg.inv(
            get_arc_horizontal_w2cs_random(
                cameras,
                look_at,
                up_direction,
                num_frames=n_steps,
                endpoint=False,
            )
        ).numpy()
        # fovs = np.full((n_steps,), fov)
        Ks = np.tile(K, (n_steps * len(camtoworld), 1, 1))

    elif option == "spiral":
        camtoworld = generate_spiral_path_random(
            cameras.numpy() @ np.diagflat([1, -1, -1, 1]),
            np.array([1, 5]),
            n_frames=n_steps,
            n_rots=random.uniform(0.5, 3),
            zrate=random.uniform(-1.0, 1.0),
            radii=[1.0, 1.0, 0.5],
            endpoint=False,
        )  # @ np.diagflat([1, -1, -1, 1]) # n_frame x 3 x 4

        camtoworld = (
            torch.linalg.inv(start_w2c.unsqueeze(1)).float(
            ) @ torch.linalg.inv(camtoworld[:1]).float() @ camtoworld.float()
        )
        camtoworld = camtoworld.numpy()
        Ks = np.tile(K, (n_steps * len(camtoworld), 1, 1))

    elif option == "lemniscate":
        camtoworld = torch.linalg.inv(
            get_lemniscate_w2cs_random(
                start_w2c,
                look_at,
                up_direction,
                n_steps,
                degree=random.uniform(30.0, 90.0),
                endpoint=False,
            )
        ).numpy()
        Ks = np.tile(K, (n_steps * len(camtoworld), 1, 1))

    elif option == "roll":
        camtoworld = torch.linalg.inv(
            get_roll_w2cs(
                start_w2c,
                look_at,
                None,
                n_steps,
                degree=360.0,
                endpoint=False,
            )
        ).numpy()
        Ks = np.tile(K, (n_steps * len(camtoworld), 1, 1))

    elif option in [
        "dollyzoom-in",
        "dollyzoom-out",
        "zoom-in",
        "zoom-out",
    ]:
        fov = calculate_fov(K[0])
        if option.startswith("dolly"):
            direction = "backward" if option == "dollyzoom-in" else "forward"
            camtoworld = torch.linalg.inv(
                get_moving_w2cs(
                    cameras,
                    look_at,
                    up_direction,
                    n_steps,
                    endpoint=True,
                    direction=direction,
                )
            ).numpy()
        else:
            camtoworld = np.tile(np.expand_dims(
                cameras, 1), (1, n_steps, 1, 1))
        fov_rad_start = fov
        if zoom_factor is None:
            zoom_factor = 0.28 if option.endswith("zoom-in") else 1.5
        fov_rad_end = zoom_factor * fov
        fovs = (
            np.linspace(0, 1, n_steps)[:, None] * (fov_rad_end - fov_rad_start)
            + fov_rad_start
        )
        Ks = get_intrinsics(fovs, K[0])
        Ks = Ks.repeat(len(camtoworld), 1, 1)
        Ks = Ks.numpy()

    elif option in [
        "move-forward",
        "move-backward",
        "move-up",
        "move-down",
        "move-left",
        "move-right",
    ]:
        camtoworld = torch.linalg.inv(
            get_moving_w2cs(
                cameras,
                look_at,
                up_direction,
                n_steps,
                endpoint=True,
                direction=option.removeprefix("move-"),
            )
        ).numpy()
        Ks = np.tile(K, (n_steps * len(camtoworld), 1, 1))

    elif option == "interp":
        camtoworld = generate_interpolated_path(
            start_w2c, n_steps,
            spline_degree=random.randint(2, 5),
            smoothness=random.uniform(0.001, 0.15),
            rot_weight=random.uniform(0.05, 0.3),
        )
        Ks = np.tile(K, (len(camtoworld), 1, 1))

    else:
        raise ValueError(f"Unknown preset option {option}.")

    return camtoworld, Ks


def generate_interpolated_path(
    poses: np.ndarray,
    n_interp: int,
    spline_degree: int = 5,
    smoothness: float = 0.03,
    rot_weight: float = 0.1,
):
    """Creates a smooth spline path between input keyframe camera poses.

    Spline is calculated with poses in format (position, lookat-point, up-point).

    Args:
      poses: (n, 4, 4) array of input pose keyframes.
      n_interp: returned path will have n_interp * (n - 1) total poses.
      spline_degree: polynomial degree of B-spline.
      smoothness: parameter for spline smoothing, 0 forces exact interpolation.
      rot_weight: relative weighting of rotation/translation in spline solve.

    Returns:
      Array of new camera poses with shape (n_interp * (n - 1), 4, 4).
    """

    def poses_to_points(poses, dist):
        """Converts from pose matrices to (position, lookat, up) format."""
        pos = poses[:, :3, -1]
        lookat = pos - dist * poses[:, :3, 2]
        up = pos + dist * poses[:, :3, 1]
        return np.stack([pos, lookat, up], 1)

    def points_to_poses(points):
        """Converts from (position, lookat, up) format to pose matrices."""
        return np.array([viewmatrix(p - l, u - p, p) for p, l, u in points])

    def interp(points, n, k, s):
        """Runs multidimensional B-spline interpolation on the input points."""
        sh = points.shape
        pts = np.reshape(points, (sh[0], -1))
        k = min(k, sh[0] - 1)
        tck, _ = scipy.interpolate.splprep(pts.T, k=k, s=s)
        u = np.linspace(0, 1, n, endpoint=False)
        new_points = np.array(scipy.interpolate.splev(u, tck))
        new_points = np.reshape(new_points.T, (n, sh[1], sh[2]))
        return new_points

    points = poses_to_points(poses, dist=rot_weight)
    new_points = interp(
        points, n_interp * (points.shape[0] - 1), k=spline_degree, s=smoothness
    )

    camtoworld = points_to_poses(new_points)
    camtoworld = np.concatenate(
        [
            camtoworld,
            np.array([0.0, 0.0, 0.0, 1.0])[
                None, None].repeat(len(camtoworld), 0),
        ],
        1,
    )
    return camtoworld


def get_arc_horizontal_w2cs(
    ref_c2w: torch.Tensor,
    lookat: torch.Tensor,
    up: torch.Tensor | None,
    num_frames: int,
    clockwise: bool = True,
    face_off: bool = False,
    endpoint: bool = False,
    degree: float = 360.0,
    ref_up_shift: float = 0.0,
    ref_radius_scale: float = 0.5,
) -> torch.Tensor:
    ref_position = ref_c2w[:, :3, 3]
    if up is None:
        up = -ref_c2w[:, :3, 1]
    elif len(up.shape) < 2:
        up = up.unsqueeze(0)

    assert up is not None

    camera_dist = torch.linalg.norm(ref_c2w[:, :3, 3] - lookat)
    shift_ratio = random.uniform(-0.5, 0.5)
    ref_up_shift = shift_ratio * camera_dist.item()

    ref_position += up * ref_up_shift
    ref_position *= ref_radius_scale
    thetas = (
        torch.linspace(0.0, torch.pi * degree / 180,
                       num_frames, device=ref_c2w.device)
        if endpoint
        else torch.linspace(
            0.0, torch.pi * degree / 180, num_frames + 1, device=ref_c2w.device
        )[:-1]
    )
    if not clockwise:
        thetas = -thetas

    vec = ref_position - lookat   # B, 3
    vec = vec.unsqueeze(1).expand(-1, num_frames, -1)
    positions = (
        torch.einsum(
            "bnij,bnj->bni",
            # B, n_frames, 3, 3
            roma.rotvec_to_rotmat(
                thetas[:, None] * np.expand_dims(up, axis=1)),
            vec,
        )
        + lookat
    )  # B, n_frames, 3
    return get_lookat_w2cs(positions, lookat, up, face_off=face_off)


def get_arc_horizontal_w2cs_random(
    ref_c2w: torch.Tensor,
    lookat: torch.Tensor,
    up: torch.Tensor | None,
    num_frames: int,
    clockwise: bool = True,
    face_off: bool = False,
    endpoint: bool = False,
    degree: float = 360.0,
    ref_up_shift: float = 0.0,
    ref_radius_scale: float = 0.5,
) -> torch.Tensor:
    B = ref_c2w.shape[0]
    device = ref_c2w.device
    # degree: [120.0, 360.0]
    degree = torch.rand(B, device=device) * 240.0 + 120.0
    # ref_radius_scale: [0.2, 0.6]
    ref_radius_scale = torch.rand(B, device=device) * 0.3 + 0.3
    # shift_ratio: [-0.5, 0.5]
    shift_ratio = torch.rand(B, device=device) * 1.0 - 0.5

    ref_position = ref_c2w[:, :3, 3]
    if up is None:
        up = -ref_c2w[:, :3, 1]
    elif len(up.shape) < 2:
        up = up.unsqueeze(0)

    if len(lookat.shape) == 1:
        current_lookat = lookat.unsqueeze(0).expand(B, -1)  # B, 3
    else:
        current_lookat = lookat

    camera_dist = torch.linalg.norm(
        ref_c2w[:, :3, 3] - current_lookat, dim=1)  # B
    ref_up_shift = shift_ratio * camera_dist

    ref_position += up * ref_up_shift.unsqueeze(1)
    ref_position *= ref_radius_scale.unsqueeze(1)
    base_thetas = (
        torch.linspace(0.0, torch.pi, num_frames, device=device)
        if endpoint
        else torch.linspace(
            0.0, torch.pi, num_frames + 1, device=device
        )[:-1]
    )
    thetas_factor = degree[:, None] / 180.0
    thetas = thetas_factor * base_thetas
    if not clockwise:
        thetas = -thetas

    vec = ref_position - current_lookat   # B, 3
    rotvec = thetas.unsqueeze(-1) * up.unsqueeze(1)  # B, n_frames, 3
    rotmat = roma.rotvec_to_rotmat(rotvec)

    # (B, 3) -> (B, 1, 3) -> (B, n_frames, 3)
    vec_expanded = vec.unsqueeze(1).expand(-1, num_frames, -1)
    positions = (
        torch.einsum(
            "bnij,bnj->bni",
            rotmat,
            vec_expanded,
        )
        + current_lookat.unsqueeze(1)  # lookat: (B, 3) -> (B, 1, 3)
    )  # B, n_frames, 3
    return get_lookat_w2cs(
        positions, lookat.unsqueeze(1), up.unsqueeze(1), face_off=face_off
    )


def normalize(x):
    """Normalization helper function."""
    return x / np.linalg.norm(x)


def viewmatrix(lookdir, up, position, subtract_position=False):
    """Construct lookat view matrix."""
    vec2 = normalize((lookdir - position) if subtract_position else lookdir)
    vec0 = normalize(np.cross(up, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, position], axis=1)
    return m


def normalize_torch(vec):
    """Normalize a vector or a batch of vectors (L2 norm)."""
    # Handles inputs like (B, n_frames, 3)
    return vec / torch.linalg.norm(vec, dim=-1, keepdim=True)


def viewmatrix_vectorized(lookdir, up, position, subtract_position=False):
    """Construct lookat view matrix using PyTorch vectorization."""
    # lookdir: (B, n_frames, 3)
    # up: (1, 1, 3) or (B, 1, 3)
    if subtract_position:
        vec2 = normalize_torch(lookdir - position)
    else:
        vec2 = normalize_torch(lookdir)
    # 2. vec0 (X-axis of camera, right vector)
    vec0 = normalize_torch(torch.cross(up, vec2, dim=-1))

    # 3. vec1 (Y-axis of camera, up vector)
    vec1 = normalize_torch(torch.cross(vec2, vec0, dim=-1))
    R = torch.stack([vec0, vec1, vec2], dim=-1)
    T = position.unsqueeze(-1)
    P_3x4 = torch.cat([R, T], dim=-1)
    last_row = torch.tensor([0., 0., 0., 1.], device=lookdir.device).view(
        1, 1, 1, 4).expand(P_3x4.shape[0], P_3x4.shape[1], -1, -1)

    m = torch.cat([P_3x4, last_row], dim=-2)  # (B, n_frames, 4, 4)

    return m


def poses_avg(poses):
    """New pose using average position, z-axis, and up vector of input poses."""
    position = poses[:, :3, 3].mean(0)
    z_axis = poses[:, :3, 2].mean(0)
    up = poses[:, :3, 1].mean(0)
    cam2world = viewmatrix(z_axis, up, position)
    return cam2world


def generate_spiral_path(
    poses, bounds, n_frames=120, n_rots=2, zrate=0.5, endpoint=False, radii=None
):
    """Calculates a forward facing spiral path for rendering."""
    # Find a reasonable 'focus depth' for this dataset as a weighted average
    # of near and far bounds in disparity space.
    close_depth, inf_depth = bounds.min() * 0.9, bounds.max() * 5.0
    dt = 0.75
    focal = 1 / ((1 - dt) / close_depth + dt / inf_depth)

    # Get radii for spiral path using 90th percentile of camera positions.
    positions = poses[:, :3, 3]
    if radii is None:
        radii = np.percentile(np.abs(positions), 90, 0)
    radii = np.concatenate([radii, [1.0]])

    # Generate poses for spiral path.
    render_poses = []
    cam2world = poses_avg(poses)
    up = poses[:, :3, 1].mean(0)
    for theta in np.linspace(0.0, 2.0 * np.pi * n_rots, n_frames, endpoint=endpoint):
        t = radii * [np.cos(theta), -np.sin(theta), -
                     np.sin(theta * zrate), 1.0]
        position = cam2world @ t
        lookat = cam2world @ [0, 0, -focal, 1.0]
        z_axis = position - lookat
        render_poses.append(viewmatrix(z_axis, up, position))
    render_poses = np.stack(render_poses, axis=0)
    return render_poses  # [frames, B, 4]


def generate_spiral_path_random(
    poses, bounds, n_frames=120, n_rots=2, zrate=0.5, endpoint=False, radii=None
):
    """Calculates a forward facing spiral path for rendering."""
    B = poses.shape[0]
    device = "cpu"
    n_rots = torch.rand(B, device=device) * 1.5 + 1.5  # [1.5, 3]
    zrate = torch.rand(B, device=device) * 1.5 - 0.5  # [-0.5, 1.0]
    # Find a reasonable 'focus depth' for this dataset as a weighted average
    # of near and far bounds in disparity space.
    close_depth, inf_depth = bounds.min() * 0.9, bounds.max() * 5.0
    dt = 0.75
    focal = 1 / ((1 - dt) / close_depth + dt / inf_depth)

    # Get radii for spiral path using 90th percentile of camera positions.
    positions = poses[:, :3, 3]
    if radii is None:
        radii = np.percentile(np.abs(positions), 90, 0)
    radii = np.concatenate([radii, [1.0]])
    radii = torch.from_numpy(radii).to(device)

    # Generate poses for spiral path.
    render_poses = []
    cam2world = poses_avg(poses)
    up = poses[:, :3, 1].mean(0)

    max_angle = 2.0 * np.pi
    thetas = (
        torch.linspace(0.0, max_angle, n_frames, device=device)
        if endpoint
        else torch.linspace(0.0, max_angle, n_frames + 1, device=device)[:-1]
    )  # (n_frames)
    n_rots_expanded = n_rots[:, None]  # B, 1
    zrate_expanded = zrate[:, None]  # B, 1
    thetas_batch = thetas[None, :] * n_rots_expanded  # B, n_frames

    cos_t = torch.cos(thetas_batch).unsqueeze(-1)  # B, n_frames, 1
    sin_t = -torch.sin(thetas_batch).unsqueeze(-1)  # B, n_frames, 1
    # B, n_frames, 1
    sin_z = -torch.sin(thetas_batch * zrate_expanded).unsqueeze(-1)
    constant_term = torch.ones_like(cos_t)
    t = torch.cat([cos_t, sin_t, sin_z, constant_term],
                  dim=-1)  # B, n_frames, 4
    t_scaled = t * radii[None, None, :]

    cam2world = torch.from_numpy(cam2world)
    cam2world_expand = torch.cat([cam2world, torch.tensor(
        [[0., 0., 0., 1.]], device=device)], dim=0)  # 4x4
    # (B, 4, 4) @ (B, 4, n_frames) = (B, 4, n_frames)
    position_h = cam2world_expand.unsqueeze(0).expand(
        B, 4, 4) @ t_scaled.transpose(-1, -2)
    position = position_h[:, :3, :].transpose(-1, -2)  # B, n_frames, 3

    # 4
    lookat_h = cam2world @ torch.tensor([0, 0, -focal, 1.0], device=device)
    lookat = lookat_h[:3]  # 3
    z_axis = position - lookat.view(1, 1, 3)
    up_expanded = torch.from_numpy(up[None, None, :])
    render_poses = viewmatrix_vectorized(z_axis, up_expanded, position)
    return render_poses


def get_lemniscate_w2cs(
    ref_w2c: torch.Tensor,
    lookat: torch.Tensor,
    up: torch.Tensor | None,
    num_frames: int,
    degree: float,
    endpoint: bool = False,
    **_,
) -> torch.Tensor:
    degree = random.uniform(30.0, 90.0),
    ref_c2w = torch.linalg.inv(ref_w2c)
    a = torch.linalg.norm(ref_c2w[:, :3, 3] - lookat) * \
        np.tan(degree / 360 * np.pi)
    # Lemniscate curve in camera space. Starting at the origin.
    thetas = (
        torch.linspace(0, 2 * torch.pi, num_frames, device=ref_w2c.device)
        if endpoint
        else torch.linspace(0, 2 * torch.pi, num_frames + 1, device=ref_w2c.device)[:-1]
    ) + torch.pi / 2
    positions = torch.stack(
        [
            a * torch.cos(thetas) / (1 + torch.sin(thetas) ** 2),
            a * torch.cos(thetas) * torch.sin(thetas) /
            (1 + torch.sin(thetas) ** 2),
            torch.zeros(num_frames, device=ref_w2c.device),
        ],
        dim=-1,
    )
    # Transform to world space.R
    positions = torch.einsum(
        "bij,nj->bni", ref_c2w[:, :3], F.pad(positions, (0, 1), value=1.0)
    )  # B, N, 3
    if up is None:
        up = -ref_c2w[:, :3, 1]
    elif len(up.shape) < 2:
        up = up.unsqueeze(0)
    assert up is not None
    return get_lookat_w2cs(positions, lookat, up)


def get_lemniscate_w2cs_random(
    ref_w2c: torch.Tensor,
    lookat: torch.Tensor,
    up: torch.Tensor | None,
    num_frames: int,
    degree: float,
    endpoint: bool = False,
    **_,
) -> torch.Tensor:
    B = ref_w2c.shape[0]
    device = ref_w2c.device
    degree = torch.rand(B, device=device) * 60.0 + 30.0  # B

    ref_c2w = torch.linalg.inv(ref_w2c)
    if len(lookat.shape) == 1:
        current_lookat = lookat.unsqueeze(0).expand(B, -1)  # B, 3
    else:
        current_lookat = lookat

    camera_dist = torch.linalg.norm(ref_c2w[:, :3, 3] - current_lookat, dim=1)
    degree_radians = degree / 360.0 * np.pi
    a = camera_dist * torch.tan(degree_radians)

    # Lemniscate curve in camera space. Starting at the origin.
    thetas = (
        torch.linspace(0, 2 * torch.pi, num_frames, device=device)
        if endpoint
        else torch.linspace(0, 2 * torch.pi, num_frames + 1, device=device)[:-1]
    ) + torch.pi / 2

    cos_t = torch.cos(thetas)
    sin_t = torch.sin(thetas)

    # positions_camera: (n_frames, 3)
    positions_camera = torch.stack(
        [
            cos_t / (1 + sin_t ** 2),
            cos_t * sin_t / (1 + sin_t ** 2),
            torch.zeros(num_frames, device=device),
        ],
        dim=-1,
    )  # N, 3

    positions_camera_scaled = a[:, None, None] * positions_camera[None, :, :]

    # Transform to world space.R
    pad_value = torch.ones_like(positions_camera_scaled[..., :1])
    positions_homo = torch.cat(
        [positions_camera_scaled, pad_value], dim=-1)  # B, N, 4

    # Transform to world space.R
    positions = torch.einsum(
        "bij,bnj->bni", ref_c2w[:, :3], positions_homo
    )  # B, N, 3
    if up is None:
        up = -ref_c2w[:, :3, 1]
    elif len(up.shape) < 2:
        up = up.unsqueeze(0).expand(B, -1)

    if len(current_lookat.shape) < 2:
        current_lookat = current_lookat.unsqueeze(0)  # 1, 3
    up_expanded = up.unsqueeze(1).expand(-1, num_frames, -1)  # B, N, 3
    return get_lookat_w2cs(positions, current_lookat.unsqueeze(1), up_expanded)


def get_roll_w2cs(
    ref_w2c: torch.Tensor,
    lookat: torch.Tensor,
    up: torch.Tensor | None,
    num_frames: int,
    endpoint: bool = False,
    degree: float = 360.0,
    **_,
) -> torch.Tensor:
    ref_c2w = torch.linalg.inv(ref_w2c)
    ref_position = ref_c2w[:, :3, 3]
    if up is None:
        up = -ref_c2w[:, :3, 1]  # Infer the up vector from the reference.

    # Create vertical angles
    thetas = (
        torch.linspace(0.0, torch.pi * degree / 180,
                       num_frames, device=ref_w2c.device)
        if endpoint
        else torch.linspace(
            0.0, torch.pi * degree / 180, num_frames + 1, device=ref_w2c.device
        )[:-1]
    )[:, None]

    lookat_vector = F.normalize(lookat[None].float(), dim=-1)[None]  # 1, 1, 3
    up = up.unsqueeze(1)
    up = (
        up * torch.cos(thetas)
        + torch.cross(lookat_vector, up) * torch.sin(thetas)
        + lookat_vector
        * torch.einsum("ij,bij->bi", lookat_vector.squeeze(0), up)[:, None]
        * (1 - torch.cos(thetas))
    )

    # Normalize the camera orientation
    ref_position = ref_position.unsqueeze(
        1).repeat(1, num_frames, 1)  # B, N, 3
    return get_lookat_w2cs(ref_position, lookat.float(), up)


def get_moving_w2cs(
    ref_c2w: torch.Tensor,
    lookat: torch.Tensor,
    up: torch.Tensor | None,
    num_frames: int,
    endpoint: bool = False,
    direction: str = "forward",
    tilt_xy: torch.Tensor = None,
):
    """
    Args:
        ref_c2w: (4, 4) tensor of the reference camera-to-world matrix
        lookat: (3,) tensor of lookat point
        up: (3,) tensor of up vector

    Returns:
        w2cs: (N, 3, 3) tensor of world to camera rotation matrices
    """
    ref_position = ref_c2w[:, :3, -1]  # [B, 3]

    if up is None:
        up = -ref_c2w[:, :3, 1]
    elif len(up.shape) == 1:
        up = up.repeat(len(ref_c2w), 1)
        # up = up.unsqueeze(0)

    direction_vectors = {
        "forward": (lookat - ref_position).clone(),
        "backward": -(lookat - ref_position).clone(),
        "up": up.clone(),
        "down": -up.clone(),
        # batch op or dim=0
        "right": torch.cross((lookat - ref_position), up, dim=1),
        "left": -torch.cross((lookat - ref_position), up, dim=1),
    }

    if direction not in direction_vectors:
        raise ValueError(
            f"Invalid direction: {direction}. Must be one of {list(direction_vectors.keys())}"
        )

    direction_vector = F.normalize(
        direction_vectors[direction], dim=1).unsqueeze(1)   # [B, 1, 3]
    # Build steps: [N] (for "fraction" of length to move along direction)
    if endpoint:
        steps = torch.linspace(0, 0.99, num_frames, device=ref_c2w.device)
    else:
        steps = torch.linspace(0, 1, num_frames + 1, device=ref_c2w.device)
    steps = steps.unsqueeze(0).unsqueeze(-1)           # [1, N, 1]
    positions = ref_position.unsqueeze(
        1) + direction_vector * steps  # [B, N, 3]

    if tilt_xy is not None:
        if tilt_xy.ndim == 2:
            tilt_xy = tilt_xy.unsqueeze(1).expand(-1, positions.shape[1], -1)
        positions[:, :, :2] += tilt_xy

    return get_lookat_w2cs(positions, lookat, up)


def get_lookat(origins: torch.Tensor, viewdirs: torch.Tensor) -> torch.Tensor:
    """Triangulate a set of rays to find a single lookat point.

    Args:
        origins (torch.Tensor): A (N, 3) array of ray origins.
        viewdirs (torch.Tensor): A (N, 3) array of ray view directions.

    Returns:
        torch.Tensor: A (3,) lookat point.
    """

    viewdirs = torch.nn.functional.normalize(viewdirs, dim=-1)
    eye = torch.eye(3, device=origins.device, dtype=origins.dtype)[None]
    # Calculate projection matrix I - rr^T
    I_min_cov = eye - (viewdirs[..., None] * viewdirs[..., None, :])
    # Compute sum of projections
    sum_proj = I_min_cov.matmul(origins[..., None]).sum(dim=-3)
    # Solve for the intersection point using least squares
    lookat = torch.linalg.lstsq(I_min_cov.sum(
        dim=-3), sum_proj).solution[..., 0]
    # Check NaNs.
    assert not torch.any(torch.isnan(lookat))
    return lookat


def get_lookat2(c2w_matrix, distance=1.0):
    origin = c2w_matrix[:, :3, 3]
    forward_direction = -c2w_matrix[:, :3, 2]
    look_at_point = origin + forward_direction * distance
    return look_at_point
