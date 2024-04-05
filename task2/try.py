import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from PIL import Image
import numpy as np
from mixup import MixUp
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
    # CIFAR-10 train set
    train_batch_size = 16
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True, num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # CIFAR-10 test set
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_set_size = len(testset)
    test_batch_size = 64 #CHANGE THIS
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False, num_workers=2)

    for x, y in trainloader:
        # Apply MixUp transformation
        mixup_inst = MixUp(sampling_method=1, alpha=0.5)
        images, labels = MixUp.mixup(mixup_inst, x, y)
        break  # Exit the loop after the first batch
    labels = torch.argmax(labels, dim=1)

    im = Image.fromarray(
        (torch.cat(images.split(1, 0), 3).squeeze() / 2 * 255 + .5 * 255).permute(1, 2, 0).numpy().astype('uint8'))
    im.save("try.png")
    print('try.png saved.')


    # # Load model to GPU
    # model = torchvision.models.vit_b_16(weights='DEFAULT').to(device)
    # criterion = torch.nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # ## train
    # for epoch in range(2):  
    #     running_loss = 0.0
    #     for i, data in enumerate(trainloader, 0):
    #         # get the inputs; data is a list of [inputs, labels]
    #         inputs, labels = data
    #         # Move input data to GPU
    #         inputs, labels = inputs.to(device), labels.to(device)
    #         mixup_inst = MixUp(sampling_method=1, alpha=0.5)
    #         mixed_images, onehot_labels = MixUp.mixup(mixup_inst, x, y)
    #         mixed_labels = torch.argmax(onehot_labels, dim=1)
    #         # zero the parameter gradients
    #         optimizer.zero_grad()

    #         # forward + backward + optimize
    #         mixed_images = mixed_images.to(device)
    #         mixed_labels = mixed_labels.to(device)
    #         mixed_outputs = model(mixed_images)
    #         mixed_outputs = mixed_outputs.to(device)
    #         loss = criterion(mixed_outputs, mixed_labels)

    #         loss.backward()
    #         optimizer.step()

    #         # print statistics
    #         running_loss += loss.item()
    #         if i % 2000 == 1999:  # print every 2000 mini-batches
    #             print('[%d, %5d] loss: %.3f' %
    #                   (epoch + 1, i + 1, running_loss / 2000))
    #             running_loss = 0.0
    #      # Printing epoch number
    #     print("Epoch: " + str(epoch + 1))
        
    #     # Computing testing accuracy
    #     accuracy = 0.0
    #     for _, test_data in enumerate(testloader, 0):
    #         test_images, test_labels = test_data
    #         test_images = test_images.to(device)
    #         test_labels = test_labels.to(device)
    #         test_outputs = model(test_images)
    #         test_predictions = torch.argmax(test_outputs, 1, keepdim=False)
    #         accuracy += torch.sum(test_predictions == test_labels)

    #     # Computing and printing testing accuracy
    #     accuracy = 100.0 * (accuracy.item()) / (test_set_size)
    #     print("Testing accuracy: " + str(accuracy) + "%")

    # print('Training done.')

    # # save trained model
    # torch.save(model.state_dict(), 'saved_model.pt')
    # print('Model saved.')


    # Load model to GPU
    model2 = torchvision.models.vit_b_16(weights='DEFAULT').to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model2.parameters(), lr=0.001)

    ## train
    for epoch in range(2):  
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # Move input data to GPU
            inputs, labels = inputs.to(device), labels.to(device)
            mixup_inst = MixUp(sampling_method=2, alpha=0.5)
            mixed_images, onehot_labels = MixUp.mixup(mixup_inst, x, y)
            mixed_labels = torch.argmax(onehot_labels, dim=1)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            mixed_images = mixed_images.to(device)
            mixed_labels = mixed_labels.to(device)
            mixed_outputs = model2(mixed_images)
            mixed_outputs = mixed_outputs.to(device)
            loss = criterion(mixed_outputs, mixed_labels)

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
         # Printing epoch number
        print("Epoch: " + str(epoch + 1))
        
        # Computing testing accuracy
        accuracy = 0.0
        for _, test_data in enumerate(testloader, 0):
            test_images, test_labels = test_data
            test_images = test_images.to(device)
            test_labels = test_labels.to(device)
            test_outputs = model2(test_images)
            test_predictions = torch.argmax(test_outputs, 1, keepdim=False)
            accuracy += torch.sum(test_predictions == test_labels)

        # Computing and printing testing accuracy
        accuracy = 100.0 * (accuracy.item()) / (test_set_size)
        print("Testing accuracy: " + str(accuracy) + "%")

    print('Training done.')

    # save trained model
    torch.save(model2.state_dict(), 'saved_model2.pt')
    print('Model 2 saved.')

# Main function call
if __name__ == '__main__':
    main()