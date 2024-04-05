import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from PIL import Image
import numpy as np
from mixup import mixup
import warnings
import itertools
from torch.optim.lr_scheduler import CosineAnnealingLR
  

# Main function
def main():

    # Check if GPU is available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    # CIFAR-10 dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    batch_size = 16
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    for x, y in trainloader:
        # Apply MixUp transformation
        images, labels = mixup(1, x, y)
        break  # Exit the loop after the first batch
    labels = torch.argmax(labels, dim=1)

    im = Image.fromarray(
        (torch.cat(images.split(1, 0), 3).squeeze() / 2 * 255 + .5 * 255).permute(1, 2, 0).numpy().astype('uint8'))
    im.save("mixup.png")
    print('mixup.png saved.')
    print('Ground truth labels:' + ' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))

    # Load model to GPU
    model = torchvision.models.vit_b_16(weights='DEFAULT').to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    ## train
    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # Move input data to GPU
            inputs, labels = inputs.to(device), labels.to(device)
            mixed_images, onehot_labels = mixup(1, inputs, labels)
            mixed_labels = torch.argmax(onehot_labels, dim=1)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            mixed_outputs = model(mixed_images)
            loss = criterion(mixed_outputs, mixed_labels)

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Training done.')

    # save trained model
    torch.save(model.state_dict(), 'saved_model.pt')
    print('Model saved.')











    # Loading CIFAR-10 test set
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_set_size = len(testset)  # Size of testing set
    test_set_batch_size = 256
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
        # Checking CUDA memory usage
        
        # Extracting hyperparameters
        alpha, learning_rate, batch_size = hyperparams
        
        # Creating DataLoader for training set
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, generator=generator)
        
        # Setting device to GPU
        device = 'cuda:0'
        
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
        print("Image Saved")
        
        # Clearing memory
        del trainloader, dataiter, images, im
        torch.cuda.empty_cache()
        
        # Creating DataLoader for training set
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, generator=generator)
        
        # Initializing SimpleViT model
        #net = SimpleViT(image_size=32, patch_size=4, num_classes=10, dim=256, depth=12, heads=8, mlp_dim=512)
        net = SimpleViT(num_classes=10)
        
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
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=2, generator=generator)

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
