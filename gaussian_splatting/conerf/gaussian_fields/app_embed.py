# pylint: disable=[E1101]

import torch
import torch.nn.functional as F

from sklearn.neighbors import NearestNeighbors
from torch import Tensor

try:
    import tinycudann as tcnn
    TCNN_AVAILABLE=True
except ImportError as e:
    print(
        f"WARNING: {e}! "
        "Please install tinycudann by: "
        "pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"
        )
    TCNN_AVAILABLE=False


class AppearanceOptModule(torch.nn.Module):
    """Appearance optimization module."""

    def __init__(
        self,
        num_images: int,
        feature_dim: int,
        embed_dim: int = 16,
        sh_degree: int = 3,
        mlp_width: int = 64,
        mlp_depth: int = 2,
        use_tcnn: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.sh_degree = sh_degree
        self.embeds = torch.nn.Embedding(num_images, embed_dim)
        self.use_tcnn = use_tcnn

        input_dim = embed_dim + feature_dim + (sh_degree + 1) ** 2
        if use_tcnn and TCNN_AVAILABLE:
            self.color_head = tcnn.Network(
                n_input_dims=input_dim,
                n_output_dims=3,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": mlp_width,
                    "n_hidden_layers": mlp_depth,
                    # Initialize the last layer to be zero so that the initial output is zero.
                    "initialization": {
                        "weights": "zero",
                        "biases": "zero",
                    }
                },
            )
        else:
            layers = []
            layers.append(torch.nn.Linear(input_dim, mlp_width))
            layers.append(torch.nn.ReLU(inplace=True))
            for _ in range(mlp_depth - 1):
                layers.append(torch.nn.Linear(mlp_width, mlp_width))
                layers.append(torch.nn.ReLU(inplace=True))
            layers.append(torch.nn.Linear(mlp_width, 3))
            self.color_head = torch.nn.Sequential(*layers)

        self._init_mlp()

    def _init_mlp(self):
        # Initialize the last layer to be zero so that the initial output is zero.
        if (not self.use_tcnn) or (not TCNN_AVAILABLE):
            torch.nn.init.zeros_(self.color_head[-1].weight)
            torch.nn.init.zeros_(self.color_head[-1].bias)

    def forward(
        self, 
        features: Tensor, 
        embed_ids: Tensor, 
        dirs: Tensor, 
        sh_degree: int,
        embed_value: float = 0.0
    ) -> Tensor:
        """Adjust appearance based on embeddings.

        Args:
            features: (N, feature_dim)
            embed_ids: (C,)
            dirs: (C, N, 3)

        Returns:
            colors: (C, N, 3)
        """
        from gsplat.cuda._torch_impl import _eval_sh_bases_fast

        C, N = dirs.shape[:2]
        # Camera embeddings
        if embed_ids is None:
            embeds = torch.ones(C, self.embed_dim, device=features.device) * embed_value
        else:
            embeds = self.embeds(embed_ids)  # [C, D2]
        embeds = embeds[:, None, :].expand(-1, N, -1)  # [C, N, D2]
        # GS features
        features = features[None, :, :].expand(C, -1, -1)  # [C, N, D1]
        # View directions
        dirs = F.normalize(dirs, dim=-1)  # [C, N, 3]
        num_bases_to_use = (sh_degree + 1) ** 2
        num_bases = (self.sh_degree + 1) ** 2
        sh_bases = torch.zeros(
            C, N, num_bases, device=features.device)  # [C, N, K]
        sh_bases[:, :, :num_bases_to_use] = _eval_sh_bases_fast(
            num_bases_to_use, dirs)
        # Get colors
        if self.embed_dim > 0:
            h = torch.cat([embeds, features, sh_bases],
                          dim=-1)  # [C, N, D1 + D2 + K]
        else:
            h = torch.cat([features, sh_bases], dim=-1)
        h = h.squeeze(0)  # [C,N,D]
        colors = self.color_head(h)
        return colors


def knn(x: Tensor, K: int = 4) -> Tensor:
    x_np = x.cpu().numpy()
    model = NearestNeighbors(n_neighbors=K, metric="euclidean").fit(x_np)
    distances, _ = model.kneighbors(x_np)
    return torch.from_numpy(distances).to(x)
