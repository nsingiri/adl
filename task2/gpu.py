import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import torch.optim as optim

# pytorch pipeline: https://medium.com/@wolframalphav1.0/easy-way-to-improve-image-classifier-performance-part-1-mixup-augmentation-with-codes-33288db92de5
# https://medium.com/@muhammad2000ammar/mastering-transfer-learning-with-pytorch-d1521f3a6a6e

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


if __name__ == '__main__':
    # Check if GPU is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
