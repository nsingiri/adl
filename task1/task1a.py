import numpy as np
import torch 
from torch import nn 
from torch.utils.data import TensorDataset, DataLoader

def polynomial_fun(w, x):
    """
    Evaluates a polynomial function given the weight vector w and an input scalar variable x.
    Args:
        w (torch.Tensor): Weight vector of size (M + 1, 1)
        x (torch.Tensor): Input scalar variables of size (N, 1)
    Returns:
        y (torch.Tensor): Output value of the polynomial function which has size (N, 1)
    """
    M = w.shape[0]
    powers = torch.arange(M, dtype=torch.float32)
    x_powers = torch.pow(x, powers)
    y = torch.matmul(x_powers, w)
    return y

def fit_polynomial_sgd_weight_regularised(x, t, M, learning_rate, minibatch_size):
    """
    Fits a polynomial function using stochastic minibatch gradient descent.

    Args:
        x (torch.Tensor): Input data points of shape (N, 1)
        t (torch.Tensor): Target values of shape (N, 1)
        M (int): Polynomial degree
        learning_rate (float): Learning rate for gradient descent
        minibatch_size (int): Size of the minibatch

    Returns:
        w_opt (torch.Tensor): Optimum weight vector of shape (M+1, 1)
    """
    num_epochs = 2000
    x_powers = torch.pow(x, torch.arange(M+1, dtype=torch.float32))
    max_powers = (torch.max(torch.abs(x_powers), axis=0)).values
    x_powers = x_powers/max_powers
    train_data = TensorDataset(x_powers, t)
    model = nn.Linear(M+1, 1, bias=False, dtype=torch.float32) 
    mse_loss = nn.MSELoss() 
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9) 
    alpha = 10**(-M)  # Weight regularization alpha
    losses = []
    epochs = []
    # Training
    for epoch in range(num_epochs):
        minibatch_data = DataLoader(train_data, batch_size=minibatch_size, shuffle=True)
        for x, y in minibatch_data:
            optimizer.zero_grad()  
            prediction = model(x)  
            # Regularization term using L2-norm
            regularisation_term = torch.sum(torch.square(torch.div(model.weight, max_powers)))
            loss = mse_loss(prediction, y) + alpha * regularisation_term
            loss.backward() 
            optimizer.step()  

        # Print loss every 10,000 epochs
        if epoch % 100 == 0:
            # Set weights that are not influential to 0 every 10,000 epochs
            flag = torch.abs(torch.div(model.weight, max_powers)) >= 1e-3
            model.weight.data = model.weight * flag
            print("Epoch: " + str(epoch) + ", MSE + L2 weight regularization loss: " + str(loss.item()))

        epochs.append(epoch)
        losses.append(loss.item())

    print("Epoch: " + str(epoch) + ", MSE + L2 weight regularization loss: " + str(loss.item()))
    
    # Set weights to 0 for those not influential
    flag = torch.abs(torch.div(model.weight, max_powers)) >= 1e-3
    model.weight.data = model.weight * flag
    w_hat = model.weight
    
    # Rescale the weight
    print("-" * 20 + "end" + "-" * 20)
    w_opt = w_hat / max_powers

    return w_opt.reshape(M+1, 1)


def main():
    
    #Use polynomial_fun (𝑀 =10, 𝐰=[1,2,3,4,5]T) to generate a training set and a test set, in the 
    #form of respectively sampled 100 and 50 pairs of 𝑥,𝑥𝜖[−20,20], and 𝑡. The observed 𝑡 values 
    #are obtained by adding Gaussian noise (standard deviation being 0.2) to 𝑦.
    
    temp1 = torch.arange(3, dtype=torch.float32)
    temp2 = torch.tensor(1, dtype=torch.float32)
    w = torch.add(temp1, temp2)
    w = w.reshape(w.shape[0], -1)
    del temp1, temp2
    w = torch.tensor([1,2,3,4,5], dtype=torch.float32).reshape(5,1)

    #training set
    x_train = 40.0*(torch.rand(100, dtype=torch.float32) - 0.5).reshape(100,1)
    y_train = polynomial_fun(w,x_train)
    noise_train = (0.2*torch.randn(100, dtype=torch.float32)).reshape(100,1)
    t_train = y_train+noise_train
    del noise_train

    #testing set
    x_test = 40.0*(torch.rand(50, dtype=torch.float32) - 0.5).reshape(50,1)
    y_test = polynomial_fun(w,x_test)
    noise_test = 0.2*torch.randn(50, dtype=torch.float32).reshape(50,1)
    t_test = y_test + noise_test
    del noise_test

    # Report the optimized 𝑀 value and the mean (and standard deviation) in difference between the model-predicted values and the underlying “true” polynomial curve
    M_max = 10 # Maximum polynomial degree allowed during fitting
    w_hat_sgd = fit_polynomial_sgd_weight_regularised(x_train, t_train, M_max, 0.5, 25)  # Batch size=25, learning rate = 0.5

    # Print optimized weight vector and optimal polynomial degree
    print("-" * 40 + "Optimized Results" + "-" * 40)
    print("Optimized weight vector:")
    print(w_hat_sgd.tolist())
    # Find the optimal polynomial degree after training
    optimal_M = torch.max(torch.nonzero(w_hat_sgd)[:,0])
    print("\nOptimized degree of the polynomial:", optimal_M.item()+1)
    print("-" * 87)

    # Compute predicted values for both training and testing sets
    y_hat_sgd_train = polynomial_fun(w_hat_sgd, x_train)
    y_hat_sgd_test = polynomial_fun(w_hat_sgd, x_test)

    # Compute difference between predicted values and true polynomial for training set
    difference = y_hat_sgd_train - y_train
    std_difference, mean_difference = torch.std_mean(difference)
    print(". \n" * 5)
    print("-" * 20 + "Difference between predicted values (on training set) and true polynomial" + "-" * 20)
    print("Mean difference: ", mean_difference.tolist())
    print("Standard deviation: ", std_difference.tolist())
    print("-" * 20 + "end" + "-" * 20)
    del difference, std_difference, mean_difference

    # Compute difference between predicted values and true polynomial for testing set
    difference = y_hat_sgd_test - y_test
    std_difference, mean_difference = torch.std_mean(difference)
    print(". \n" * 5)
    print("-" * 20 + "Difference between predicted values (on testing set) and true polynomial" + "-" * 20)
    print("Mean difference: ", mean_difference.tolist())
    print("Standard deviation: ", std_difference.tolist())
    print("-" * 20 + "end" + "-" * 20)
    del difference, std_difference, mean_difference



if __name__=="__main__":
    main()
