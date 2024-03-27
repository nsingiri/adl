import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from PIL import Image


# Convert tensor to numpy array and normalize
mixed_x_np = mixed_x.cpu().numpy()  # Assuming mixed_x is on GPU, so move it to CPU first
mixed_x_np = (mixed_x_np * 0.5 + 0.5) * 255  # Scale to [0, 255]

# Convert tensor of labels to list
mixed_y_list = mixed_y.tolist()

# Create a blank canvas for the montage
montage = Image.new('RGB', (320, 320), color='white')

# Resize and paste each image into the montage
for i in range(16):
    img = mixed_x_np[i].transpose(1, 2, 0).astype(np.uint8)  # Convert (C, H, W) to (H, W, C)
    img_pil = Image.fromarray(img)
    img_pil = img_pil.resize((80, 80), Image.ANTIALIAS)  # Resize the image to fit the montage
    montage.paste(img_pil, (80 * (i % 4), 80 * (i // 4)))

# Draw labels on the montage
draw = ImageDraw.Draw(montage)
font = ImageFont.load_default()  # You can also specify your own font
for i, label in enumerate(mixed_y_list):
    draw.text((80 * (i % 4) + 5, 80 * (i // 4) + 5), str(label), fill='black', font=font)

# Save the montage as a PNG file
montage.save("mixup.png")

print("MixUp visualization saved as mixup.png")







# if __name__ == '__main__':
#     ## cifar-10 dataset
#     transform = transforms.Compose(
#         [transforms.ToTensor(),
#          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#
#     batch_size = 20
#     trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
#     trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
#     classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#
#     # example images
#     dataiter = iter(trainloader)
#     images, labels = next(dataiter)
#
#     im = Image.fromarray(
#         (torch.cat(images.split(1, 0), 3).squeeze() / 2 * 255 + .5 * 255).permute(1, 2, 0).numpy().astype('uint8'))
#     im.save("train_pt_images.jpg")
#     print('train_pt_images.jpg saved.')
#     print('Ground truth labels:' + ' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))
#
#
#
#
#     ## cnn
#     net = Net()
#
#     ## loss and optimiser
#     criterion = torch.nn.CrossEntropyLoss()
#     optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
#
#     ## train
#     for epoch in range(2):  # loop over the dataset multiple times
#
#         running_loss = 0.0
#         for i, data in enumerate(trainloader, 0):
#             # get the inputs; data is a list of [inputs, labels]
#             inputs, labels = data
#
#             # zero the parameter gradients
#             optimizer.zero_grad()
#
#             # forward + backward + optimize
#             outputs = net(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#
#             # print statistics
#             running_loss += loss.item()
#             if i % 2000 == 1999:  # print every 2000 mini-batches
#                 print('[%d, %5d] loss: %.3f' %
#                       (epoch + 1, i + 1, running_loss / 2000))
#                 running_loss = 0.0
#
#     print('Training done.')
#
#     # save trained model
#     torch.save(net.state_dict(), 'saved_model.pt')
#     print('Model saved.')


# for epoch in range(2):  # loop over the dataset multiple times
#     print('epoch', epoch)
#     running_loss = 0.0
#     for i, data in enumerate(dataloader, 0):
#         inputs, outputs = data
#         images, labels = mixup(1, inputs, outputs)
#         labels = torch.argmax(labels, dim=1)
#         # zero the parameter gradients
#         optimizer.zero_grad()
#
#         # forward + calculate loss
#         # if use_mix_up:
#         X_m, y_true_a, y_true_b, lam = mixup(inputs, labels, )
#         y_pred = model(images)
#
#         loss = mixup_criterion(criterion, y_pred, y_true_a, y_true_b, lam)
#         # outputs = net(inputs)
#         # loss = criterion(outputs, labels)
#         # backward + optimize
#         loss.backward()
#
#         optimizer.step()
#
#         # print statistics
#         running_loss += loss.item()
#         if i % 2000 == 1999:  # print every 2000 mini-batches
#             print('[%d, %5d] loss: %.3f' %
#                   (epoch + 1, i + 1, running_loss / 2000))
#             running_loss = 0.0


