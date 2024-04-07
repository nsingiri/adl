import torch
import torch.nn as nn
import torchvision.models as models

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.vit_b_16(weights='DEFAULT')

    def __init__(self, num_layers=6, patch_size=8, num_heads=8,hidden_dim=384, mlp_dim=1536):
        super().__init__()
        self.num_layers = num_layers
        self.patch_size = patch_size 
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.model = models.vit_b_16(weights='DEFAULT')

    def forward(self, x):
        return self.model(x)