import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from PIL import Image
from mixup_class import MixUp
from vit_model import Net


# Main function
def main():
    # Clearing GPU cache
    torch.cuda.empty_cache()
    # Choosing device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Setting device and default tensor type based on availability of CUDA
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
        torch.set_default_tensor_type(torch.cuda.FloatTensor if device.type == 'cuda' else torch.FloatTensor)
    # Setting device to GPU
    device = 'cuda:0'
    torch.set_default_tensor_type(torch.FloatTensor)

    # CIFAR-10 dataset preprocessing
    transform = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # CIFAR-10 training set
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True, num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # CIFAR-10 test set
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_set_size = len(testset) 
    test_set_batch_size = 36
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_set_batch_size, shuffle=False, num_workers=2)
    
    
    dataiter = iter(trainloader)
    images, labels = next(dataiter)
    
    # Mixup augmentation
    Mixer = MixUp(0.5, 1)
    images, _ = Mixer.mixup_fn(images, labels)
    
    # Saving augmented images
    im = Image.fromarray((torch.cat(images.split(1, 0), 3).squeeze() / 2 * 255 + .5 * 255).permute(1, 2, 0).numpy().astype('uint8'))
    im.save("mixup.png")
    
    # Clearing memory
    del trainloader, dataiter, images, im
    torch.cuda.empty_cache()
    
    # Creating DataLoader for training set
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

    # ViT model
    model = Net().to(device)
    model.num_layers = 2  
    model.patch_size = 4  
    model.hidden_dim = 16  
    model.mlp_dim = 32  
    model.num_heads = 4 
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # # Sampling Method 1
    print("Sampling method 1")
    print(device)

    # Shows good performance at 2 epochs 
    for epoch in range(20):
        for _, data in enumerate(trainloader, 0):
            # Applying mixup augmentation
            Mixer = MixUp(0.5, 1) # alpha = 0.5, sampling_method = 1
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            mixed_inputs, mixed_one_hot_labels = Mixer.mixup_fn(inputs, labels)
            mixed_labels = torch.argmax(mixed_one_hot_labels, dim=1)
            optimizer.zero_grad()
            outputs = model(mixed_inputs)
            loss = criterion(outputs, mixed_labels)
            loss.backward()
            optimizer.step()

        # Epoch number
        print("Epoch: " + str(epoch + 1))
        
        # Testing accuracy
        test_accuracy = 0.0
        for _, test_data in enumerate(testloader, 0):
            test_images, test_labels = test_data
            test_images = test_images.to(device)
            test_labels = test_labels.to(device)
            test_outputs = model(test_images)
            test_predictions = torch.argmax(test_outputs, 1, keepdim=False)
            test_accuracy += torch.sum(test_predictions == test_labels)

        # Printing testing accuracy
        test_accuracy = 100.0 * (test_accuracy.item()) / (test_set_size)
        print("Testing accuracy: " + str(test_accuracy) + "%")

    # Printing completion message after training
    print('Training done.')

    # Saving trained model
    torch.save(model.state_dict(), 'method_one_model.pt')
    print('Model one saved.')


    # Sampling Method 2
    print("Sampling method 2")
    print(device)

    # Vit Model
    model2 = Net().to(device)
    model2.num_layers = 2  
    model2.patch_size = 4  
    model2.hidden_dim = 16  
    model2.mlp_dim = 32  
    model2.num_heads = 4
    model2.heads = torch.nn.Sequential(torch.nn.Linear(in_features=768, out_features=10, bias=True))
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model2.parameters(), lr=0.0001)

    # Shows good performance at 2 epochs
    for epoch in range(20):
        for _, data in enumerate(trainloader, 0):
            # Mixup augmentation
            Mixer = MixUp(0.5, 2) # alpha = 0.5, sampling_method = 2
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            mixed_inputs, mixed_one_hot_labels = Mixer.mixup_fn(inputs, labels)
            mixed_labels = torch.argmax(mixed_one_hot_labels, dim=1)
            optimizer.zero_grad()
            outputs = model2(mixed_inputs)
            loss = criterion(outputs, mixed_labels)
            loss.backward()
            optimizer.step()

        # Epoch number
        print("Epoch: " + str(epoch + 1))
        
        # Testing accuracy
        test_accuracy = 0.0
        for _, test_data in enumerate(testloader, 0):
            test_images, test_labels = test_data
            test_images = test_images.to(device)
            test_labels = test_labels.to(device)
            test_outputs = model2(test_images)
            test_predictions = torch.argmax(test_outputs, 1, keepdim=False)
            test_accuracy += torch.sum(test_predictions == test_labels)

        # Computing and printing testing accuracy
        test_accuracy = 100.0 * (test_accuracy.item()) / (test_set_size)
        print("Testing accuracy: " + str(test_accuracy) + "%")

    # Printing completion message after training
    print('Training done.')

    # Saving trained model
    torch.save(model2.state_dict(), 'method_two_model.pt')
    print('Model two saved.')

    # Save test images
    testloader = torch.utils.data.DataLoader(testset, batch_size=36, shuffle=True, num_workers=2)
    dataiter = iter(testloader)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    images, labels = next(dataiter)

    # Save images
    im = Image.fromarray((torch.cat(images.split(1, 0), 3).squeeze() / 2 * 255 + .5 * 255).permute(1, 2, 0).numpy().astype('uint8'))
    im.save("result.png")

    images = images.to(device)
    labels = labels.to(device)

    # Printing ground-truth labels
    print('Ground-truth:\n', ' '.join('%5s' % classes[labels[j]] for j in range(36)))

    # Output of sample method one
    outputs = model(images)
    predicted = torch.argmax(outputs, 1, keepdim=False)
    print('Predicted classes of sampling method 1:\n', ' '.join('%5s' % classes[predicted[j]] for j in range(36)))

    # Output of sample method two
    outputs = model2(images)
    predicted = torch.argmax(outputs, 1, keepdim=False)
    print('Predicted classes of sampling method 2:\n', ' '.join('%5s' % classes[predicted[j]] for j in range(36)))

# Main function call
if __name__ == '__main__':
    main()
