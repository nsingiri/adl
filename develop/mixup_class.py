# Importing necessary libraries
import torch
import numpy as np

# Class definition for mixup augmentation
class mixup:
    def __init__(self, alpha, sampling_method):
        # Initializing mixup augmentation parameters
        self.alpha = alpha
        self.sampling_method = sampling_method

    def mix(self, x, y):
        # Applying mixup augmentation
        
        # Sampling lambda value based on the selected method
        if(self.sampling_method==1):
            l = np.random.beta(self.alpha,self.alpha)
        if(self.sampling_method==2):
            l = np.random.uniform(0,0.5) #range 0 to 0.5

        # Randomly shuffling indices
        indices = torch.randperm(x.shape[0]).to(x.device)
        
        # Mixed input and label tensors
        mixed_x = l*x + (1 - l)*x[indices,:,:,:]
        mixed_y = l*y + (1 - l)*y[indices]

        return mixed_x, mixed_y
