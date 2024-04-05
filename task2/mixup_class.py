import torch
import numpy as np

class mixup:
    """
    Data augmentation class that implements the mixup algorithm.
    """
    def __init__(self, alpha, sampling_method):
        """
        Set alpha parameter and get value for sampling_method (either equals 1 or 2)
        """
        self.alpha = alpha
        self.sampling_method = sampling_method

    def mixup_fn(self, images, labels):
        """
        Implement mixup algorithm.
        """
        if self.sampling_method == 1:
            lam = np.random.beta(self.alpha, self.alpha)
        if self.sampling_method == 2:
            lam = np.random.uniform(0.0, 1.0)
        input_batch_size = images.size()[0]
        index = torch.randperm(input_batch_size)
        x1, x2 = images, images[index, :]
        label = torch.nn.functional.one_hot(labels, num_classes=10)
        y1, y2 = label, label[index]
        x = lam * x1 + (1 - lam) * x2
        y = lam * y1 + (1 - lam) * y2
        return x, y
    


