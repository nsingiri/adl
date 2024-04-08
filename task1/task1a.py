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
    num_epochs = 5000
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
            reg_term = torch.sum(torch.square(torch.div(model.weight, max_powers)))
            loss = mse_loss(prediction, y) + alpha * reg_term
            loss.backward() 
            optimizer.step()  
        epochs.append(epoch)
        losses.append(loss.item())
        # Print loss every 200 epochs
        if epoch % 200 == 0:
            # Set weights that are not influential to 0 every 10,000 epochs
            flag = torch.abs(torch.div(model.weight, max_powers)) >= 1e-3
            model.weight.data = model.weight * flag
            print('Epoch: {} Loss {}'.format(epoch + 1, loss.item()))

    print('End of Training')
    print('Epoch: {} Loss {}'.format(epoch + 1, loss.item()))
    
    # Set weights to 0 for those not influential
    flag = torch.abs(torch.div(model.weight, max_powers)) >= 1e-3
    model.weight.data = model.weight * flag
    weight = model.weight
    w_opt = weight / max_powers
    return w_opt.reshape(M+1, 1)


def main():

    # Define weight vector
    w = torch.tensor([1, 2, 3], dtype=torch.float32).reshape(3, 1)

    # Generate training set
    x_train = 40.0 * (torch.rand(20, dtype=torch.float32) - 0.5).reshape(20, 1)
    y_train = polynomial_fun(w, x_train)
    noise_train = (0.5 * torch.randn(20, dtype=torch.float32)).reshape(20, 1)
    t_train = y_train + noise_train

    # Generate test set
    x_test = 40.0 * (torch.rand(10, dtype=torch.float32) - 0.5).reshape(10, 1)
    y_test = polynomial_fun(w, x_test)
    noise_test = 0.5 * torch.randn(10, dtype=torch.float32).reshape(10, 1)
    t_test = y_test + noise_test

    # Find optimized M value
    M_max = 6 # Max polynomial degree 
    w_hat_sgd = fit_polynomial_sgd_weight_regularised(x_train, t_train, M_max, 0.1, 10)  # Batch size=10, learning rate = 0.1
    optimal_M = torch.max(torch.nonzero(w_hat_sgd)[:,0])
    print("\nOptimized degree:", optimal_M.item()+1)

    # Predicted values for both training and testing sets
    y_hat_sgd_train = polynomial_fun(w_hat_sgd, x_train)
    y_hat_sgd_test = polynomial_fun(w_hat_sgd, x_test)

    # Difference between predicted values and true polynomial for training set
    difference = y_hat_sgd_train - y_train
    std_difference, mean_difference = torch.std_mean(difference)
    print("Difference between true and predicted values for train set")
    print("Mean difference (Train): ", mean_difference.tolist())
    print("Standard deviation (Train): ", std_difference.tolist())
    del difference, std_difference, mean_difference

    # Difference between predicted values and true polynomial for testing set
    difference = y_hat_sgd_test - y_test
    std_difference, mean_difference = torch.std_mean(difference)
    print("Difference between true and predicted values for test set")
    print("Mean difference (Test): ", mean_difference.tolist())
    print("Standard deviation (Test): ", std_difference.tolist())
    del difference, std_difference, mean_difference


if __name__=="__main__":
    main()
