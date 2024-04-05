import torch
import torchvision.models as models
import torch.nn as nn

class SimpleViT(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.vit_b_16 = models.vit_base_patch16_224(pretrained=True)
        self.vit_b_16.head = nn.Linear(self.vit_b_16.head.in_features, num_classes)

    def forward(self, x):
        return self.vit_b_16(x)