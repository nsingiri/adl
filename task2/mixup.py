# import torch
# import numpy as np

# class MixUp:
#     """
#     Data augmentation class that implements the mixup algorithm.
#     """
#     # def __init__(self, alpha, sampling_method):
#     #     """
#     #     Set alpha parameter and get value for sampling_method (either equals 1 or 2)
#     #     """
#     #     self.alpha = alpha
#     #     self.sampling_method = sampling_method

#     # def mixup(self, x, y):
#     #     """

#     #     """
#     #     if(self.sampling_method==1):
#     #         lam = np.random.beta(self.alpha, self.alpha)
#     #     elif(self.sampling_method==2):
#     #         lam = np.random.uniform(0, 0.5) 
#     #     # Randomly shuffling indices
#     #     indices = torch.randperm(x.shape[0]).to(x.device)
        
#     #     # Mixed input and label tensors
#     #     mixup_x = lam * x + (1 - lam) * x[indices,:,:,:]
#     #     mixup_y = lam * y + (1 - lam) * y[indices]

#     #     return mixup_x, mixup_y
    

#     # def mixup(sampling_method, images, labels):
#     #     if sampling_method == 1:
#     #         alpha = 1.0
#     #         lam = np.random.beta(alpha, alpha)
#     #     elif sampling_method == 2:
#     #         lam = np.random.uniform(0.0, 1.0)
#     #     input_batch_size = images.size()[0]
#     #     index = torch.randperm(input_batch_size)
#     #     x1, x2 = images, images[index, :]
#     #     label = torch.nn.functional.one_hot(labels, num_classes=10)
#     #     y1, y2 = label, label[index]
#     #     x = lam * x1 + (1 - lam) * x2
#     #     y = lam * y1 + (1 - lam) * y2
#     #     return x, y


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

    


