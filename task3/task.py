import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from PIL import Image
from simple_vit import SimpleViT
from mixup_class import MixUp
import time


def compute_precision(y_true, y_pred):
    """
    Compute precision

    Parameters:
        y_true (Tensor): True labels.
        y_pred (Tensor): Predicted labels.

    Returns:
        float: precision
    """
    true_positives = torch.sum((y_true == 1) & (y_pred == 1)).float()
    false_positives = torch.sum((y_true == 0) & (y_pred == 1)).float()
    precision = true_positives / (true_positives + false_positives + 1e-9)
    return precision.item() 

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
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Loading CIFAR-10 datasets
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    dataset = torch.utils.data.ConcatDataset([trainset,testset])
    length_dataset = len(dataset)
    development_set_size = int(0.8 * length_dataset)
    holdout_test_set_size = length_dataset - development_set_size
    development_set, holdout_set = torch.utils.data.random_split(dataset,[development_set_size, holdout_test_set_size])
    del dataset, length_dataset

    # Split the development set into train and validation
    train_set_size = int(0.9 * development_set_size)
    validation_set_size = development_set_size - train_set_size
    train_set, validation_set = torch.utils.data.random_split(development_set,[train_set_size, validation_set_size])
    del development_set, development_set_size
    test_set_size=holdout_test_set_size

    trainloader = torch.utils.data.DataLoader(train_set, batch_size=36, shuffle=True, num_workers=2,)
    validationloader = torch.utils.data.DataLoader(validation_set, batch_size=36, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(holdout_set, batch_size=36, shuffle=True, num_workers=2)
    
    torch.cuda.empty_cache()

    # SimpleViT model
    model = SimpleViT(grid_size=32, patch_size=4, num_classes=10, hid_channels=256, depth=12, heads=8, mlp_channels=512).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Sampling Method 1
    print("Sampling method 1")
    print(device)

    # Shows good performance at 10 epochs 
    model1_start_time = time.time()
    for epoch in range(10):
        for _, data in enumerate(trainloader, 0):
            # Applying mixup augmentation
            Mixer = MixUp(0.5, 1) # alpha = 0.5, sampling_method = 1
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            mixed_inputs, mixed_one_hot_labels = Mixer.mixup_fn(inputs, labels)
            
            optimizer.zero_grad()
            outputs = model(mixed_inputs)
            loss = criterion(outputs, mixed_one_hot_labels)
            loss.backward()
            optimizer.step()

        # Epoch number
        print("Epoch: " + str(epoch + 1))

        v_accuracy = 0.0
        mse_loss_net1_val=0.0
        model1_precision = 0.0
        with torch.no_grad():
            for _, validation_data in enumerate(validationloader, 0):
                validation_images, validation_labels = validation_data
                validation_images = validation_images.to(device)
                validation_labels = validation_labels.to(device)
                validation_outputs = model(validation_images)
                validation_predictions = torch.argmax(validation_outputs, 1, keepdim=False)
                v_accuracy += torch.sum(validation_predictions == validation_labels)
                validation_labels_one_hot = torch.nn.functional.one_hot(validation_labels, num_classes=10).float()
                mse_loss_net1_val += torch.nn.functional.mse_loss(validation_outputs, validation_labels_one_hot.float(), reduction='sum')
                model1_precision += compute_precision(validation_labels, validation_predictions)
            #cross_entropy_net1_val += torch.sum(torch.log(val_softmax_output[torch.arange(val_softmax_output.shape[0]),validation_labels]))
        # Computing and printing testing accuracy
        v_accuracy = 100.0 * (v_accuracy.item()) / (validation_set_size)
        print("Validation set accuracy: " + str(v_accuracy) + "%")
        mse_loss_net1_val = 100.0 * (mse_loss_net1_val.item()) / (validation_set_size)
        print("Validation set MSE loss: {:.4f}".format(mse_loss_net1_val)+ "%")
        model1_precision /= len(validationloader)
        print("Precision: {:.4f}".format(model1_precision))
        
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

    model1_time_elapsed =  time.time() - model1_start_time
    # Saving trained model
    torch.save(model.state_dict(), 'sample_one_task_three.pt')
    print('Model saved.')



    # Sampling Method 2
    print("Sampling method 2")
    print(device)

    # Initializing SimpleViT model for sampling method 2
    model2 = SimpleViT(grid_size=16, patch_size=4, num_classes=10, hid_channels=256, depth=6, heads=8, mlp_channels=512).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model2.parameters(), lr=0.0001)

    # Shows good performance at 10 epochs
    model2_start_time = time.time()
    for epoch in range(10):
        for _, data in enumerate(trainloader, 0):
            # Mixup augmentation
            Mixer = MixUp(0.5, 2) # alpha = 0.5, sampling_method = 2
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            mixed_inputs, mixed_one_hot_labels = Mixer.mixup_fn(inputs, labels)
            optimizer.zero_grad()
            outputs = model2(mixed_inputs)
            loss = criterion(outputs, mixed_one_hot_labels)
            loss.backward()
            optimizer.step()

        # Epoch number
        print("Epoch: " + str(epoch + 1))

        v_accuracy2 = 0.0
        mse_loss_net2_val=0.0
        model2_precision = 0.0
        with torch.no_grad():
          for _, validation_data in enumerate(validationloader, 0):
            validation_images, validation_labels = validation_data
            validation_images = validation_images.to(device)
            validation_labels = validation_labels.to(device)
            validation_outputs = model2(validation_images)
            validation_predictions = torch.argmax(validation_outputs, 1, keepdim=False)
            v_accuracy2 += torch.sum(validation_predictions == validation_labels)
            validation_labels_one_hot = torch.nn.functional.one_hot(validation_labels, num_classes=10).float()
            mse_loss_net2_val += torch.nn.functional.mse_loss(validation_outputs, validation_labels_one_hot.float(), reduction='sum')
            model2_precision += compute_precision(validation_labels, validation_predictions)
                #cross_entropy_net1_val += torch.sum(torch.log(val_softmax_output[torch.arange(val_softmax_output.shape[0]),validation_labels]))
        # Computing and printing testing accuracy
        v_accuracy2 = 100.0 * (v_accuracy2.item()) / (validation_set_size)
        print("Validation set accuracy: " + str(v_accuracy) + "%")
        mse_loss_net2_val = 100.0 * (mse_loss_net2_val.item()) / (validation_set_size)
        print("Validation set MSE loss: {:.4f}".format(mse_loss_net2_val)+ "%")
        model2_precision /= len(validationloader)
        print("Precision: {:.4f}".format(model2_precision))
        
        # Testing accuracy
        test_accuracy2 = 0.0
        for _, test_data in enumerate(testloader, 0):
            test_images, test_labels = test_data
            test_images = test_images.to(device)
            test_labels = test_labels.to(device)
            test_outputs = model2(test_images)
            test_predictions = torch.argmax(test_outputs, 1, keepdim=False)
            test_accuracy2 += torch.sum(test_predictions == test_labels)

        # Computing and printing testing accuracy
        test_accuracy2 = 100.0 * (test_accuracy2.item()) / (test_set_size)
        print("Testing accuracy: " + str(test_accuracy2) + "%")
    
    model2_time_elapsed =  time.time() - model2_start_time
    # Saving trained model
    torch.save(model2.state_dict(), 'sample_two_task_three.pt')
    print('Model 2 saved.')
    

    print("\nSummary of metrics for the networks with sampling method 1 and 2")
    print(f"{'Metric' : <60}{'Network 1' : <30}{'Network 2' : <30}")
    print(f"{'Validation Accuracy' : <60}{v_accuracy:.2f}{'%' : <28}{v_accuracy2:.2f}{'%' : <28}")
    print(f"{'Validation Loss' : <60}{mse_loss_net1_val:.2f}{'%' : <28}{mse_loss_net2_val:.2f}{'%' : <28}")
    print(f"{'Holdout set Accuracy' : <60}{test_accuracy:.2f}{'%' : <28}{test_accuracy2:.2f}{'%' : <28}")
    print(f"{'Time to train (seconds)' : <60}{model1_time_elapsed:.2f}{'s' : <28}{model2_time_elapsed:.2f}{'s' : <28}")
    print(f"{'F1 Score' : <60}{model1_precision:.4f}{' ' : <28}{model2_precision:.4f}")


# Main function call
if __name__ == '__main__':
    main()
