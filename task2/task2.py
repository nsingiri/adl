import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.optim as optim
from PIL import Image


# set random seed
torch.manual_seed(21)


class MixUp:
    """The MixUp class implements the mixup data augmentation algorithm"""
    def __init__(self, sampling_method):
        super().__init__()
        # self.vit.head = nn.Linear(self.vit.head.in_features, num_classes)  # Change the head to output num_classes
        self.sampling_method = sampling_method
        self.model = torch.torchvision.models.vit_b_16(weights='DEFAULT')

    def forward(self, x):
        return self.model(x)

    @staticmethod
    def mixup(sampling_method, loader1, loader2):
        # model = torch.torchvision.models.vit_b_16(weights='DEFAULT')
        if sampling_method == 1:
            alpha = np.random.uniform(0, 10.0)
            lam = np.random.beta(alpha, alpha)
        elif sampling_method == 2:
            lam = np.random.uniform(0.0, 1.0)
        for (x1, y1), (x2, y2) in zip(loader1, loader2):
            x = lam * x1 + (1 - lam) * x2
            y = lam * y1 + (1 - lam) * y2
        return x, y


if __name__ == '__main__':
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 16

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    # Define the size of each subset
    subset1_size = len(trainset) // 2
    subset2_size = len(trainset) - subset1_size

    # Split the dataset into two subsets
    subset1, subset2 = torch.utils.data.random_split(trainset, [subset1_size, subset2_size])

    # Create data loaders for each subset
    batch_size = 16
    subset1_loader = torch.utils.data.DataLoader(subset1, batch_size=batch_size, shuffle=True, num_workers=2)
    subset2_loader = torch.utils.data.DataLoader(subset2, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # example images
    # dataiter = iter(trainloader)
    # images, labels = next(dataiter)
    mixed_loader = MixUp.mixup(sampling_method=1, loader1=subset1_loader, loader2=subset2_loader)

    # Assuming you have already defined or imported the mixed_loader
    # Assuming batch_size is known
    batch_size = 16

    # Get a batch of mixed images and labels
    mixed_data = next(iter(mixed_loader))
    mixed_images, mixed_labels = mixed_data[0], mixed_data[1]

    # Randomly select 16 images from the batch
    selected_indices = np.random.choice(batch_size, 16, replace=False)
    selected_images = mixed_images[selected_indices]
    selected_labels = mixed_labels[selected_indices]

    # Create a grid of images
    grid_image = torchvision.utils.make_grid(selected_images, nrow=4)

    # Convert the grid image to a PIL image
    pil_image = transforms.ToPILImage()(grid_image)

    # Save the PIL image as a JPEG file
    pil_image.save("montage.jpg")

    # Print the ground truth labels of the selected images
    print('Ground truth labels: ' + ' '.join('%5s' % classes[selected_labels[j]] for j in range(16)))



    # Get the mixed images and labels
    # dataiter = iter(mixed_loader)
    # images, labels = next(dataiter)
    #
    # # Print the mixed images and their ground truth labels
    # im = Image.fromarray(
    #     (torch.cat(images.split(1, 0), 3).squeeze() / 2 * 255 + .5 * 255).permute(1, 2, 0).numpy().astype('uint8'))
    # im.save("montage.png")
    #
    # print('Mixed images saved.')
    # print('Ground truth labels: ' + ' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))

    # Get the mixed images and labels
    # dataiter = iter(mixed_loader)
    # images, labels = next(dataiter)
    #
    # im = Image.fromarray(
    #     (torch.cat(images.split(1, 0), 3).squeeze() / 2 * 255 + .5 * 255).permute(1, 2, 0).numpy().astype('uint8'))
    # im.save("montage.png")
    # print('train_pt_images.jpg saved.')
    # print('Ground truth labels:' + ' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))



    # model = MixUp(sampling_method=1)
    # criterion = torch.nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    #
    #
    # # Training loop
    # for epoch in range(2):  # 20 epochs
    #     running_loss = 0.0
    #     for i, data in enumerate(trainloader, 0):
    #         inputs, labels = data
    #         # inputs, targets_a, targets_b, lam = mixup(inputs, labels)
    #
    #         optimizer.zero_grad()
    #         outputs = net(inputs)
    #         # loss = criterion(outputs, labels)
    #         loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
    #         loss.backward()
    #         optimizer.step()
    #
    #         # Print statistics
    #         running_loss += loss.item()
    #         if i % 2000 == 1999:  # Print every 2000 mini-batches
    #             print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
    #             running_loss = 0.0
    #
    # print('Training done.')
    # # Save trained model
    # torch.save(net.state_dict(), 'vit_model.pt')
    # print('Model saved.')

    # def test():
    #     dataiter = iter(testloader)
    #     classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    #
    #     ## load the trained model
    #     model = Net()
    #     model.load_state_dict(torch.load('vit_model.pt'))
    #
    #     ## inference
    #     images, labels = next(dataiter)
    #     print('Ground-truth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
    #
    #     outputs = model(images)
    #     _, predicted = torch.max(outputs, 1)
    #     print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))
    #
    #     # save to images
    #     im = Image.fromarray(
    #         (torch.cat(images.split(1, 0), 3).squeeze() / 2 * 255 + .5 * 255).permute(1, 2, 0).numpy().astype('uint8'))
    #     im.save("test_pt_images.jpg")
    #     print('test_pt_images.jpg saved.')
