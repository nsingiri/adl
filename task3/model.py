import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def mixup(sampling_method, images, labels):
    if sampling_method == 1:
        alpha = 1.0
        lam = np.random.beta(alpha, alpha)
    elif sampling_method == 2:
        lam = np.random.uniform(0.0, 1.0)
    input_batch_size = images.size()[0]
    index = torch.randperm(input_batch_size)
    x1, x2 = images, images[index, :]
    label = torch.nn.functional.one_hot(labels, num_classes=10)
    y1, y2 = label, label[index]
    x = lam * x1 + (1 - lam) * x2
    y = lam * y1 + (1 - lam) * y2
    return x, y


class Net(nn.Module):
    def __init__(self, sampling_method):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

