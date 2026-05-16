# pylint: disable=[E1101]

import torch


def strip_lower_diagonal(L: torch.Tensor):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device=L.device)

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 1]

    return uncertainty


def strip_symmetric(symmetric_mat: torch.Tensor):
    return strip_lower_diagonal(symmetric_mat)


def normalize_quaternion(quaternion: torch.Tensor):
    """Normalize quaternions.

    Args:
        quaternion (B, 4): quaternions using the Hamilton notation

    Return:
        Normalized quaternion.
    """
    norm = torch.sqrt(
        quaternion[:, 0] * quaternion[:, 0] +
        quaternion[:, 1] * quaternion[:, 1] +
        quaternion[:, 2] * quaternion[:, 2] +
        quaternion[:, 3] * quaternion[:, 3]
    )

    normalized_quat = quaternion / norm[:, None]
    return normalized_quat


def quaternion_to_rotation_mat(quaternion: torch.Tensor):
    """ Convert quaternions to rotation matrices.

    Args:
        quaternion (B, 4): quaternions using the Hamilton notation.
    """
    normalized_quaternion = normalize_quaternion(quaternion)
    R = torch.zeros((quaternion.size(0), 3, 3), device=quaternion.device)

    r = normalized_quaternion[:, 0]
    x = normalized_quaternion[:, 1]
    y = normalized_quaternion[:, 2]
    z = normalized_quaternion[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)

    return R


def rotation_mat_left_multiply_scale_mat(
    scale_mat: torch.Tensor,
    quaternion: torch.Tensor,
) -> torch.Tensor:
    """Construct a matrix by left multiplying the scale matrices by rotation matrices.

    Args:
        scale_mat (B, 3, 3): scaling matrices.
        quaternion (B, 4): quaternions using the Hamilton notation.

    Return:
        the constructed matrices L = R @ S
    """

    L = torch.zeros(
        (scale_mat.shape[0], 3, 3), dtype=torch.float, device=quaternion.device
    )
    R = quaternion_to_rotation_mat(quaternion)

    L[:, 0, 0] = scale_mat[:, 0]
    L[:, 1, 1] = scale_mat[:, 1]
    L[:, 2, 2] = scale_mat[:, 2]

    L = R @ L
    return L
