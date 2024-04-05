# Importing necessary libraries
import torch
from torch import nn

from einops import rearrange
from einops.layers.torch import Rearrange  # Importing necessary functions from einops library

# Helper functions

# Function to ensure that dimensions are pairs
def pair(t):
    return t if isinstance(t, tuple) else (t, t)  # If input is a tuple, return it; otherwise, create a tuple with the input value repeated twice

# Function to generate positional embeddings using sine and cosine functions
def posemb_sincos_2d(patches, temperature=10000, dtype=torch.float32):
    _, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    # Create meshgrid of y and x coordinates
    y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')

    # Calculate frequencies for sine and cosine functions
    assert (dim % 4) == 0, 'feature dimension must be multiple of 4 for sincos emb'
    omega = torch.arange(dim // 4, device=device) / (dim // 4 - 1)
    omega = 1. / (temperature ** omega)

    # Calculate positional embeddings using sine and cosine functions
    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)

    return pe.type(dtype)  # Return positional embeddings with the specified data type

# Classes

# Feed-forward neural network module
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        # Define feed-forward network layers
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),  # GELU activation function
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)  # Forward pass through the network

# Attention mechanism module
class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads  # Inner dimension for multi-head attention
        self.heads = heads
        self.scale = dim_head ** -0.5  # Scaling factor
        self.norm = nn.LayerNorm(dim)  # Layer normalization

        self.attend = nn.Softmax(dim=-1)  # Softmax activation for attention weights

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)  # Linear transformation for queries, keys, and values
        self.to_out = nn.Linear(inner_dim, dim, bias=False)  # Linear transformation for output

    def forward(self, x):
        x = self.norm(x)  # Normalize input

        qkv = self.to_qkv(x).chunk(3, dim=-1)  # Split into queries, keys, and values
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)  # Reshape for multi-head attention

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # Compute dot products

        attn = self.attend(dots)  # Apply softmax to obtain attention weights

        out = torch.matmul(attn, v)  # Compute weighted sum of values
        out = rearrange(out, 'b h n d -> b n (h d)')  # Reshape output
        return self.to_out(out)  # Linear transformation for output

# Transformer module
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):  # Create multiple layers of attention and feed-forward networks
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head),
                FeedForward(dim, mlp_dim)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x  # Apply attention mechanism and add residual connection
            x = ff(x) + x  # Apply feed-forward network and add residual connection
        return x

# Vision Transformer model
class SimpleViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=3, dim_head=64):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        # Convert image patches into embeddings
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b h w (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)  # Transformer encoder

        self.to_latent = nn.Identity()  # Identity function
        self.linear_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)  # Linear layer for classification
        )

    def forward(self, img):
        *_, h, w, dtype = *img.shape, img.dtype

        x = self.to_patch_embedding(img)  # Convert image to patches and project to embedding space
        pe = posemb_sincos_2d(x)  # Generate positional embeddings
        x = rearrange(x, 'b ... d -> b (...) d') + pe  # Add positional embeddings to patch embeddings

        x = self.transformer(x)  # Transformer encoder
        x = x.mean(dim=1)  # Average over patch dimension

        x = self.to_latent(x)  # Identity operation
        return self.linear_head(x)  # Classification head
