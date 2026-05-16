import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
from functools import partial
from typing import Callable, Optional, Tuple
from .transformer import MLP, RMSNorm, init_weights

try:
    import xformers.ops as xops
    device_capability = torch.cuda.get_device_capability("cuda")
    is_sm80_or_sm90 = device_capability in [(8, 0), (9, 0), (10, 0), (12, 0)]
    if is_sm80_or_sm90:
        OP_TUP = (xops.fmha.flash.FwOp, xops.fmha.flash.BwOp)
    else:
        OP_TUP = None
        if torch.distributed.get_rank() == 0:
            print(f"--- [Warning]: we are using the default xformers attention kernel without flash attention. Since our capability is {device_capability} < (8, 0)")
except ImportError:
    raise ImportError("Please install xformers to use flashatt v2")

# Add these PRoPE utility functions to transformer.py
def _rope_precompute_coeffs(
    positions: torch.Tensor,  # (seqlen,)
    freq_base: float,
    freq_scale: float,
    feat_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute RoPE coefficients."""
    assert len(positions.shape) == 1
    assert feat_dim % 2 == 0
    num_freqs = feat_dim // 2
    freqs = freq_scale * (
        freq_base
        ** (
            -torch.arange(num_freqs, device=positions.device)[None, None, None, :]
            / num_freqs
        )
    )
    angles = positions[None, None, :, None] * freqs
    # Shape should be: `(batch, num_heads, seqlen, num_freqs)`; we're
    # broadcasting across `batch` and `num_heads`.
    assert angles.shape == (1, 1, positions.shape[0], num_freqs)
    return torch.cos(angles), torch.sin(angles)


def _rope_apply_coeffs(
    feats: torch.Tensor,  # (batch, num_heads, seqlen, feat_dim)
    coeffs: Tuple[torch.Tensor, torch.Tensor],
    inverse: bool = False,
) -> torch.Tensor:
    """Apply RoPE coefficients to features. We adopt a 'split' ordering
    convention. (in contrast to 'interleaved')"""
    cos, sin = coeffs
    assert len(feats.shape) == len(cos.shape) == len(sin.shape) == 4
    assert cos.shape[-1] == sin.shape[-1] == feats.shape[-1] // 2
    x_in = feats[..., : feats.shape[-1] // 2]
    y_in = feats[..., feats.shape[-1] // 2 :]
    return torch.cat(
        (
            [cos * x_in + sin * y_in, -sin * x_in + cos * y_in]
            if not inverse
            else [cos * x_in - sin * y_in, sin * x_in + cos * y_in]
        ),
        dim=-1,
    )


def _apply_block_diagonal(
    feats: torch.Tensor,  # (..., dim)
    func_size_pairs: list[tuple[Callable[[torch.Tensor], torch.Tensor], int]],
) -> torch.Tensor:
    """Apply a block-diagonal function to an input array.

    Each function is specified as a tuple with form:

        ((Tensor) -> Tensor, int)

    Where the integer is the size of the input to the function.
    """
    funcs, block_sizes = zip(*func_size_pairs)
    assert feats.shape[-1] == sum(block_sizes)
    x_blocks = torch.split(feats, block_sizes, dim=-1)
    out = torch.cat(
        [f(x_block) for f, x_block in zip(funcs, x_blocks)],
        dim=-1,
    )
    assert out.shape == feats.shape, "Input/output shapes should match."
    return out


def _apply_tiled_projmat(
    feats: torch.Tensor,  # (batch, num_heads, seqlen, feat_dim)
    matrix: torch.Tensor,  # (batch, cameras, D, D)
) -> torch.Tensor:
    """Apply projection matrix to features."""
    # - seqlen => (cameras, patches_x * patches_y)
    # - feat_dim => (feat_dim // 4, 4)
    (batch, num_heads, seqlen, feat_dim) = feats.shape
    cameras = matrix.shape[1]
    assert seqlen > cameras and seqlen % cameras == 0
    D = matrix.shape[-1]
    assert matrix.shape == (batch, cameras, D, D)
    assert feat_dim % D == 0
    return torch.einsum(
        "bcij,bncpkj->bncpki",
        matrix,
        feats.reshape((batch, num_heads, cameras, -1, feat_dim // D, D)),
    ).reshape(feats.shape)


def _invert_SE3(transforms: torch.Tensor) -> torch.Tensor:
    """Invert a 4x4 SE(3) matrix."""
    assert transforms.shape[-2:] == (4, 4)
    Rinv = transforms[..., :3, :3].transpose(-1, -2)
    out = torch.zeros_like(transforms)
    out[..., :3, :3] = Rinv
    out[..., :3, 3] = -torch.einsum("...ij,...j->...i", Rinv, transforms[..., :3, 3])
    out[..., 3, 3] = 1.0
    return out


def _lift_K(Ks: torch.Tensor) -> torch.Tensor:
    """Lift 3x3 matrices to homogeneous 4x4 matrices."""
    assert Ks.shape[-2:] == (3, 3)
    out = torch.zeros(Ks.shape[:-2] + (4, 4), device=Ks.device)
    out[..., :3, :3] = Ks
    out[..., 3, 3] = 1.0
    return out


def _invert_K(Ks: torch.Tensor) -> torch.Tensor:
    """Invert 3x3 intrinsics matrices. Assumes no skew."""
    assert Ks.shape[-2:] == (3, 3)
    out = torch.zeros_like(Ks)
    out[..., 0, 0] = 1.0 / Ks[..., 0, 0]
    out[..., 1, 1] = 1.0 / Ks[..., 1, 1]
    out[..., 0, 2] = -Ks[..., 0, 2] / Ks[..., 0, 0]
    out[..., 1, 2] = -Ks[..., 1, 2] / Ks[..., 1, 1]
    out[..., 2, 2] = 1.0
    return out


class ProPEAttention(nn.Module):
    """
    PRoPE-enabled attention that can fall back to standard attention.
    Integrates with existing flash attention infrastructure.
    """

    def __init__(
        self,
        dim,
        head_dim,
        qkv_bias=False,
        fc_bias=True,
        attn_dropout=0.0,
        fc_dropout=0.0,
        use_qk_norm=True,
        # PRoPE-specific parameters
        cameras=None,
        patches_x=None,
        patches_y=None,
        image_width=None,
        image_height=None,
        freq_base=100.0,
        freq_scale=1.0,
    ):
        """
        Args:
            dim: Input dimension
            head_dim: Dimension of each attention head
            qkv_bias: Whether to use bias in QKV projection
            fc_bias: Whether to use bias in output projection
            attn_dropout: Dropout probability for attention weights
            fc_dropout: Dropout probability for output projection
            use_qk_norm: Whether to use Q-K normalization
            cameras: Number of cameras (for PRoPE)
            patches_x: Number of patches in x direction (for PRoPE)
            patches_y: Number of patches in y direction (for PRoPE)
            image_width: Image width for intrinsics normalization (for PRoPE)
            image_height: Image height for intrinsics normalization (for PRoPE)
            freq_base: RoPE frequency base
            freq_scale: RoPE frequency scale
        """
        super().__init__()
        assert dim % head_dim == 0, f"Token dimension {dim} should be divisible by head dimension {head_dim}"
        
        self.dim = dim
        self.head_dim = head_dim
        self.num_heads = dim // head_dim
        self.attn_dropout = attn_dropout
        self.use_qk_norm = use_qk_norm

        # Standard attention components
        self.to_qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.fc = nn.Linear(dim, dim, bias=fc_bias)
        self.attn_fc_dropout = nn.Dropout(fc_dropout)
        
        # Optional Q-K normalization
        if self.use_qk_norm:
            self.q_norm = RMSNorm(head_dim)
            self.k_norm = RMSNorm(head_dim)

        # PRoPE-specific parameters
        self.cameras = cameras
        self.patches_x = patches_x
        self.patches_y = patches_y
        self.image_width = image_width
        self.image_height = image_height
        self.freq_base = freq_base
        self.freq_scale = freq_scale

        # Precompute PRoPE coefficients if parameters are provided
        if all(x is not None for x in [cameras, patches_x, patches_y]):
            self._precompute_prope_coeffs()

    def _precompute_prope_coeffs(self):
        """Precompute PRoPE RoPE coefficients."""
        coeffs_x = _rope_precompute_coeffs(
            torch.tile(torch.arange(self.patches_x), (self.patches_y * self.cameras,)),
            freq_base=self.freq_base,
            freq_scale=self.freq_scale,
            feat_dim=self.head_dim // 4,
        )
        coeffs_y = _rope_precompute_coeffs(
            torch.tile(
                torch.repeat_interleave(torch.arange(self.patches_y), self.patches_x),
                (self.cameras,),
            ),
            freq_base=self.freq_base,
            freq_scale=self.freq_scale,
            feat_dim=self.head_dim // 4,
        )
        # Do not save coeffs to checkpoint as `cameras` might change during testing.
        self.register_buffer("coeffs_x_0", coeffs_x[0], persistent=False)
        self.register_buffer("coeffs_x_1", coeffs_x[1], persistent=False)
        self.register_buffer("coeffs_y_0", coeffs_y[0], persistent=False)
        self.register_buffer("coeffs_y_1", coeffs_y[1], persistent=False)

    def _prope_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        viewmats: torch.Tensor,
        Ks: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Apply PRoPE attention with flash attention backend."""
        batch, num_heads, seqlen, head_dim = q.shape
        cameras = viewmats.shape[1]
        
        assert q.shape == k.shape == v.shape
        assert viewmats.shape == (batch, cameras, 4, 4)
        assert Ks is None or Ks.shape == (batch, cameras, 3, 3)
        assert seqlen == cameras * self.patches_x * self.patches_y
        assert head_dim % 4 == 0

        # Normalize camera intrinsics if provided
        if Ks is not None:
            Ks_norm = torch.zeros_like(Ks)
            Ks_norm[..., 0, 0] = Ks[..., 0, 0] / self.image_width
            Ks_norm[..., 1, 1] = Ks[..., 1, 1] / self.image_height
            Ks_norm[..., 0, 2] = Ks[..., 0, 2] / self.image_width - 0.5
            Ks_norm[..., 1, 2] = Ks[..., 1, 2] / self.image_height - 0.5
            Ks_norm[..., 2, 2] = 1.0

            # Compute projection matrices
            P = torch.einsum("...ij,...jk->...ik", _lift_K(Ks_norm), viewmats)
            P_T = P.transpose(-1, -2)
            P_inv = torch.einsum(
                "...ij,...jk->...ik",
                _invert_SE3(viewmats),
                _lift_K(_invert_K(Ks_norm)),
            )
        else:
            # GTA formula - P is `camera<-world` transform
            P = viewmats
            P_T = P.transpose(-1, -2)
            P_inv = _invert_SE3(viewmats)

        assert P.shape == P_inv.shape == (batch, cameras, 4, 4)
        # Get precomputed coefficients
        coeffs_x = (self.coeffs_x_0, self.coeffs_x_1)
        coeffs_y = (self.coeffs_y_0, self.coeffs_y_1)
        # coeffs_x, coeffs_y = self._get_prope_coeffs(seqlen, q.device)

        # Define block-diagonal transforms
        transforms_q = [
            (partial(_apply_tiled_projmat, matrix=P_T), head_dim // 2),
            (partial(_rope_apply_coeffs, coeffs=coeffs_x), head_dim // 4),
            (partial(_rope_apply_coeffs, coeffs=coeffs_y), head_dim // 4),
        ]
        transforms_kv = [
            (partial(_apply_tiled_projmat, matrix=P_inv), head_dim // 2),
            (partial(_rope_apply_coeffs, coeffs=coeffs_x), head_dim // 4),
            (partial(_rope_apply_coeffs, coeffs=coeffs_y), head_dim // 4),
        ]
        transforms_o = [
            (partial(_apply_tiled_projmat, matrix=P), head_dim // 2),
            (partial(_rope_apply_coeffs, coeffs=coeffs_x, inverse=True), head_dim // 4),
            (partial(_rope_apply_coeffs, coeffs=coeffs_y, inverse=True), head_dim // 4),
        ]

        # Apply transforms and compute attention
        q_transformed = _apply_block_diagonal(q, transforms_q)
        k_transformed = _apply_block_diagonal(k, transforms_kv)
        v_transformed = _apply_block_diagonal(v, transforms_kv)

        # Use flash attention if available
        try:
            # Convert to xformers format: (batch, seqlen, num_heads, head_dim)
            q_xf = rearrange(q_transformed, "b nh l dh -> b l nh dh")
            k_xf = rearrange(k_transformed, "b nh l dh -> b l nh dh")
            v_xf = rearrange(v_transformed, "b nh l dh -> b l nh dh")
            
            out = xops.memory_efficient_attention(
                q_xf, k_xf, v_xf,
                attn_bias=None,
                p=self.attn_dropout if self.training else 0.0,
                op=OP_TUP,
            )
            
            # Convert back to (batch, num_heads, seqlen, head_dim)
            out = rearrange(out, "b l nh dh -> b nh l dh")
        except:
            # Fallback to standard scaled dot product attention
            out = F.scaled_dot_product_attention(
                q_transformed, k_transformed, v_transformed,
                dropout_p=self.attn_dropout if self.training else 0.0,
                **kwargs
            )

        # Apply output transforms
        out = _apply_block_diagonal(out, transforms_o)
        return out

    def forward(
        self,
        x,
        pos_enc="none",
        viewmats=None,
        Ks=None,
        attn_bias=None,
        **kwargs
    ):
        """
        Forward pass with configurable positional encoding.
        
        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            pos_enc: Positional encoding type ("prope", "gta", "none")
            viewmats: View matrices for PRoPE/GTA (batch, cameras, 4, 4)
            Ks: Intrinsic matrices for PRoPE (batch, cameras, 3, 3)
            attn_bias: Optional attention bias mask
            
        Returns:
            Output tensor of shape (batch, seq_len, dim)
        """
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        
        # Reshape to (batch, num_heads, seq_len, head_dim)
        q, k, v = (rearrange(t, "b l (nh dh) -> b nh l dh", dh=self.head_dim) for t in (q, k, v))
        
        # Apply qk normalization if enabled
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Apply appropriate attention based on pos_enc
        if pos_enc == "prope":
            assert viewmats is not None, "viewmats required for PRoPE"
            assert Ks is not None, "Ks required for PRoPE"
            x = self._prope_attention(q, k, v, viewmats=viewmats, Ks=Ks, **kwargs)
        elif pos_enc == "gta":
            assert viewmats is not None, "viewmats required for GTA"
            # GTA is effectively PRoPE without intrinsics
            x = self._prope_attention(q, k, v, viewmats=viewmats, Ks=None, **kwargs)
        elif pos_enc == "none":
            # Use flash attention if available, otherwise fall back to standard attention
            try:
                x = xops.memory_efficient_attention(
                    q, k, v,
                    attn_bias=attn_bias,
                    p=self.attn_dropout if self.training else 0.0,
                    op=OP_TUP,
                )
            except:
                # Fallback to standard attention
                x = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=attn_bias,
                    dropout_p=self.attn_dropout if self.training else 0.0,
                    **kwargs
                )
        else:
            raise ValueError(f"Invalid pos_enc: {pos_enc}")
        
        # Reshape back and apply output projection
        x = rearrange(x, "b nh l dh -> b l (nh dh)")
        x = self.attn_fc_dropout(self.fc(x))
        
        return x


class ProPETransformerBlock(nn.Module):
    """
    Transformer block with PRoPE attention support.
    """

    def __init__(
        self,
        dim,
        head_dim,
        ln_bias=False,
        attn_qkv_bias=False,
        attn_dropout=0.0,
        attn_fc_bias=False,
        attn_fc_dropout=0.0,
        mlp_ratio=4,
        mlp_bias=False,
        mlp_dropout=0.0,
        use_qk_norm=True,
        # PRoPE-specific parameters
        cameras=None,
        patches_x=None,
        patches_y=None,
        image_width=None,
        image_height=None,
        freq_base=100.0,
        freq_scale=1.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, bias=ln_bias)
        self.attn = ProPEAttention(
            dim=dim,
            head_dim=head_dim,
            qkv_bias=attn_qkv_bias,
            fc_bias=attn_fc_bias,
            attn_dropout=attn_dropout,
            fc_dropout=attn_fc_dropout,
            use_qk_norm=use_qk_norm,
            cameras=cameras,
            patches_x=patches_x,
            patches_y=patches_y,
            image_width=image_width,
            image_height=image_height,
            freq_base=freq_base,
            freq_scale=freq_scale,
        )

        self.norm2 = nn.LayerNorm(dim, bias=ln_bias)
        self.mlp = MLP(
            dim=dim,
            mlp_ratio=mlp_ratio,
            bias=mlp_bias,
            dropout=mlp_dropout,
        )

    def forward(self, x, pos_enc="none", viewmats=None, Ks=None, **kwargs):
        """
        Forward pass with configurable positional encoding.
        
        Args:
            x: Input tensor
            pos_enc: Positional encoding type ("prope", "gta", "none")
            viewmats: View matrices for PRoPE/GTA
            Ks: Intrinsic matrices for PRoPE
        """
        x = x + self.attn(
            self.norm1(x),
            pos_enc=pos_enc,
            viewmats=viewmats,
            Ks=Ks,
            **kwargs
        )
        x = x + self.mlp(self.norm2(x))
        return x