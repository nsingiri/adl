# Import necessary libraries
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from PIL import Image
import numpy as np
from vision import SimpleViT
#from deepvit import DeepViT
from mixup_class import mixup
import warnings
import itertools
from torch.optim.lr_scheduler import CosineAnnealingLR

# Function to check CUDA memory
def check_cuda_memory(device_id=0):
    """
    Function to check CUDA memory usage on a specified device.

    Parameters:
    - device_id (int): Index of the CUDA device to be checked (default is 0).

    Prints:
    - Device Name: Name of the CUDA device.
    - Total Memory: Total memory of the CUDA device in gigabytes (GB).
    - Memory Allocated: Memory currently allocated on the CUDA device in GB.
    - Memory Reserved: Memory reserved on the CUDA device in GB.
    """
    # Check if CUDA is available
    if torch.cuda.is_available():
        # Get device properties
        device = torch.device(f'cuda:{device_id}')
        properties = torch.cuda.get_device_properties(device)
        
        # Print device properties
        print(f'Device Name: {properties.name}')
        print(f'Total Memory: {properties.total_memory / (1024**3):.2f} GB')
        
        # Print memory currently allocated and memory reserved
        print(f'Memory Allocated: {torch.cuda.memory_allocated(device) / (1024**3):.2f} GB')
        print(f'Memory Reserved: {torch.cuda.memory_reserved(device) / (1024**3):.2f} GB')
    else:
        print('CUDA is not available. Make sure you have installed the necessary drivers.')


# Main function
def main():
    # Checking for GPU and clearing GPU cache
    torch.cuda.empty_cache()

    # Choosing device (GPU if available, otherwise CPU)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Setting device and default tensor type based on availability of CUDA
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
        torch.set_default_tensor_type(torch.cuda.FloatTensor if device.type == 'cuda' else torch.FloatTensor)
    
    # Setting device to GPU
    device = 'cuda:0'
    
    # Checking CUDA memory usage
    check_cuda_memory(device_id=0)
    
    # Setting device to CPU
    device = torch.device('cpu')

    # Setting default tensor type to CPU tensor
    torch.set_default_tensor_type(torch.FloatTensor)

    # Initializing generator on CPU
    generator = torch.Generator(device)
    
    # Setting seed for reproducibility
    generator.manual_seed(np.random.randint(0, 1000))

    # CIFAR-10 dataset preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Loading CIFAR-10 training set
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Loading CIFAR-10 test set
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_set_size = len(testset)  # Size of testing set
    test_set_batch_size = 36
    # Creating DataLoader for test set
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_set_batch_size, shuffle=False, num_workers=2)

    # Hyperparameters grid for grid search
    hyperparams_grid = {
        'alpha': [0.4],
        'learning_rate': [0.0005],
        'batch_size': [64]
    }

    # Generating all combinations of hyperparameters
    hyperparams_combinations = list(itertools.product(*hyperparams_grid.values()))
    
    # Looping through each hyperparameter combination
    for hyperparams in hyperparams_combinations:
        check_cuda_memory(device_id=0)
        # Extracting hyperparameters
        alpha, learning_rate, batch_size = hyperparams
        
        # Creating DataLoader for training set
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True, num_workers=2)
        
        
        # Creating DataLoader iterator
        dataiter = iter(trainloader)
        
        # Extracting images and labels for a batch
        images, labels = next(dataiter)
        
        # Applying mixup augmentation
        sampling_method=1
        Mixer = mixup(alpha, sampling_method)
        images, _ = Mixer.mix(images, labels)
        
        # Saving augmented images
        im = Image.fromarray((torch.cat(images.split(1, 0), 3).squeeze() / 2 * 255 + .5 * 255).permute(1, 2, 0).numpy().astype('uint8'))
        im.save("mixup.png")
        
        # Clearing memory
        del trainloader, dataiter, images, im
        torch.cuda.empty_cache()
        
        # Creating DataLoader for training set
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
        
        # Initializing SimpleViT model
        net = SimpleViT(image_size=32, patch_size=4, num_classes=10, dim=256, depth=12, heads=8, mlp_dim=512)
        print(net)
        
        # Moving model to GPU
        net.to(device)
        
        # Defining loss function
        criterion = torch.nn.MSELoss()

        # Initializing AdamW optimizer with specified learning rate
        optimizer = optim.AdamW(net.parameters(), lr=learning_rate)

        # Maximum number of epochs for training
        Max_Epochs = 20

        # Setting sampling method for mixup augmentation
        sampling_method = 1

        # Printing start message
        print("-" * 20 + "start" + "-" * 20)

        # Printing information about mixup augmentation and device
        print("Sampling method 1 (alpha value: " + str(alpha) + ")\nwith pre-trained Vit initialization")
        print(device)

        # Printing hyperparameters
        print(alpha, learning_rate, batch_size, Max_Epochs)

        # Looping through epochs for training
        for epoch in range(Max_Epochs):
            running_loss = 0.0
            
            # Iterating through batches in training loader
            for i, data in enumerate(trainloader, 0):
                # Applying mixup augmentation
                Mixer = mixup(alpha, sampling_method)
                inputs, labels = data
                one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=10)
                inputs = inputs.to(device)
                one_hot_labels = 1.0 * one_hot_labels.to(device)
                mixed_inputs, mixed_one_hot_labels = Mixer.mix(inputs, one_hot_labels)
                
                # Zeroing gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = net(mixed_inputs)
                
                # Calculating loss
                loss = criterion(outputs, mixed_one_hot_labels)
                
                # Backpropagation
                loss.backward()
                
                # Optimizer step
                optimizer.step()

            # Printing epoch number
            print("Epoch: " + str(epoch + 1))
            
            # Computing testing accuracy
            accuracy = 0.0
            for _, test_data in enumerate(testloader, 0):
                test_images, test_labels = test_data
                test_images = test_images.to(device)
                test_labels = test_labels.to(device)
                test_outputs = net(test_images)
                test_predictions = torch.argmax(test_outputs, 1, keepdim=False)
                accuracy += torch.sum(test_predictions == test_labels)

            # Computing and printing testing accuracy
            accuracy = 100.0 * (accuracy.item()) / (test_set_size)
            print("Testing accuracy: " + str(accuracy) + "%")

        # Printing completion message after training
        print('Training done.')

        # Saving trained model
        torch.save(net.state_dict(), 'sampling_method_one_model.pt')
        print('Model saved.')


    # Setting sampling method for mixup augmentation to 2
    sampling_method = 2

    # Printing start message
    print("-" * 20 + "start" + "-" * 20)

    # Printing information about mixup augmentation method and pre-trained ResNet initialization
    print("Sampling method 2 with pre-trained Vit initialization.\nLamda values sampled uniformly from [0,0.5)")

    # Setting hyperparameters
    alpha = 0.4
    learning_rate = 0.0005
    batch_size = 64
    Max_Epochs = 20

    # Defining loss function
    criterion = torch.nn.MSELoss()

    # Initializing SimpleViT model for sampling method 2
    net2 = SimpleViT(image_size=16, patch_size=4, num_classes=10, dim=256, depth=6, heads=8, mlp_dim=512)

    # Moving model to specified device
    net2.to(device)

    # Initializing AdamW optimizer for training
    optimizer = optim.AdamW(net2.parameters(), lr=learning_rate)

    # Looping through epochs for training
    for epoch in range(Max_Epochs):
        running_loss = 0.0
        
        # Iterating through batches in training loader
        for i, data in enumerate(trainloader, 0):
            # Applying mixup augmentation
            Mixer = mixup(alpha, sampling_method)
            inputs, labels = data
            one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=10)
            inputs = inputs.to(device)
            one_hot_labels = 1.0 * one_hot_labels.to(device)
            mixed_inputs, mixed_one_hot_labels = Mixer.mix(inputs, one_hot_labels)
            
            # Zeroing gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = net2(mixed_inputs)
            
            # Calculating loss
            loss = criterion(outputs, mixed_one_hot_labels)
            
            # Backpropagation
            loss.backward()
            
            # Optimizer step
            optimizer.step()

        # Printing epoch number and device
        print("Epoch: " + str(epoch + 1), device)
        
        # Computing testing accuracy
        accuracy = 0.0
        for _, test_data in enumerate(testloader, 0):
            test_images, test_labels = test_data
            test_images = test_images.to(device)
            test_labels = test_labels.to(device)
            test_outputs = net2(test_images)
            test_predictions = torch.argmax(test_outputs, 1, keepdim=False)
            accuracy += torch.sum(test_predictions == test_labels)

        # Computing and printing testing accuracy
        accuracy = 100.0 * (accuracy.item()) / (test_set_size)
        print("Testing accuracy: " + str(accuracy) + "%")

    # Printing completion message after training
    print('Training done.')

    # Saving trained model
    torch.save(net2.state_dict(), 'sampling_method_two_model.pt')
    print('Model saved.')

    # Visualizing results by saving to a PNG file "result.png", a montage of 36 test images with
# printed messages clearly indicating the ground-truth and the predicted classes for each.

    # Creating DataLoader for test set
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=2)

    # Creating iterator for test DataLoader
    dataiter = iter(testloader)

    # Class labels for CIFAR-10 dataset
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Inference
    images, labels = next(dataiter)

    # Save images
    im = Image.fromarray((torch.cat(images.split(1, 0), 3).squeeze() / 2 * 255 + .5 * 255).permute(1, 2, 0).numpy().astype('uint8'))
    im.save("result.png")

    # Moving images and labels to specified device
    images = images.to(device)
    labels = labels.to(device)

    # Printing ground-truth labels
    print('Ground-truth:\n', ' '.join('%5s' % classes[labels[j]] for j in range(36)))

    # Forward pass through first network (trained with sampling method 1)
    outputs = net(images)
    predicted = torch.argmax(outputs, 1, keepdim=False)
    print('Predicted by network trained with sampling method 1:\n', ' '.join('%5s' % classes[predicted[j]] for j in range(36)))

    # Forward pass through second network (trained with sampling method 2)
    outputs = net2(images)
    predicted = torch.argmax(outputs, 1, keepdim=False)
    print('Predicted by network trained with sampling method 2:\n', ' '.join('%5s' % classes[predicted[j]] for j in range(36)))

# Main function call
if __name__ == '__main__':
    main()
