import torch
import torchvision
import torch.nn as nn

class SimpleViT(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.vit_b_16 = torchvision.models.vit_b_16(weights='DEFAULT')
        #self.vit_b_16.head = nn.Linear(self.vit_b_16.head.in_features, num_classes)

    def forward(self, x):
        return self.vit_b_16(x)