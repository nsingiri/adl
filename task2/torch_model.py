from typing import Tuple
import torch
from torch import nn


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def posemb_sincos_2d(patches: torch.Tensor,
                     temperature: int = 10000,
                     dtype: torch.dtype = torch.float32) -> torch.Tensor:
    # _, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype
    h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device))
    assert (dim % 4) == 0, 'feature dimension must be multiple of 4 for sincos emb'
    omega = torch.arange(dim // 4, device=device) / (dim // 4 - 1)
    omega = 1. / (temperature**omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


class FeedForward(nn.Module):
    def __init__(self, in_channels: int, hid_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, hid_channels),
            nn.GELU(),
            nn.Linear(hid_channels, in_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, hid_channels: int, heads: int = 8, head_channels: int = 64):
        super().__init__()
        inner_channels = head_channels * heads
        self.heads = heads
        self.scale = head_channels**-0.5
        self.norm = nn.LayerNorm(hid_channels)

        self.attend = nn.Softmax(dim=-1)

        self.to_qkv = nn.Linear(hid_channels, inner_channels * 3, bias=False)
        self.to_out = nn.Linear(inner_channels, hid_channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)

        qkv = self.to_qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        q = q.view(q.size(0), -1, self.heads, q.size(-1) // self.heads)
        k = k.view(k.size(0), -1, self.heads, k.size(-1) // self.heads)
        v = v.view(v.size(0), -1, self.heads, v.size(-1) // self.heads)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = out.view(out.size(0), -1, self.heads * (v.size(-1)))
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, hid_channels, depth, heads, head_channels, mlp_channels):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    Attention(hid_channels, heads=heads, head_channels=head_channels),
                    FeedForward(hid_channels, mlp_channels)
                ]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x    


class SimpleViT(nn.Module):

    def __init__(self,
                 grid_size,
                 patch_size,
                 hid_channels,
                 depth,
                 heads,
                 mlp_channels,
                 num_classes,
                 in_channels=3,
                 head_channels=64):
        super().__init__()
        grid_height, grid_width = pair(grid_size)
        patch_height, patch_width = pair(patch_size)

        assert grid_height % patch_height == 0 and grid_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (grid_height // patch_height) * (grid_width // patch_width)
        patch_channels = in_channels * patch_height * patch_width

        self.patch_height = patch_height
        self.patch_width = patch_width

        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels, hid_channels, kernel_size=(patch_height, patch_width), stride=(patch_height, patch_width)),
        )

        self.transformer = Transformer(hid_channels, depth, heads, head_channels, mlp_channels)

        self.to_latent = nn.Identity()
        self.linear_head = nn.Sequential(nn.LayerNorm(hid_channels), nn.Linear(hid_channels, num_classes))

        # Define positional embeddings layer
        self.pe = nn.Parameter(torch.randn(num_patches, hid_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        *_, h, w, dtype = *x.shape, x.dtype

        x = self.to_patch_embedding(x)
        x = x.view(-1, self.patch_height * self.patch_width, x.size(1))
        pe = self.pe.unsqueeze(0)  # Reshape positional embeddings
        x = x + pe.expand(x.size(0), -1, -1)  # Add positional embeddings to each patch

        x = self.transformer(x)
        x = x.mean(dim=1)

        x = self.to_latent(x)
        return self.linear_head(x)
