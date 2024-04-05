import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from PIL import Image


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


def fit_polynomial_ls(x, t, M):
    """ 
    Implement a least squares solver for fitting polynomial functions using PyTorch's linear algebra modules.
    
    Args:
    - x (torch.Tensor): Input data points of shape (N, 1)
    - t (torch.Tensor): Target values of shape (N, 1)
    - M (int): Polynomial degree
    
    Returns:
    - w_hat (torch.Tensor): Optimum weight vector of shape (M+1, 1)
    """
    x_powers = torch.pow(x, torch.arange(M+1, dtype=torch.float32))
    w_hat = torch.linalg.lstsq(x_powers, t).solution
    return w_hat


def fit_polynomial_sgd(x, t, M, learning_rate, minibatch_size):
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
    losses = []
    epochs = []
    # Training
    for epoch in range(num_epochs):
        minibatch_data = DataLoader(train_data, batch_size=minibatch_size, shuffle=True)
        for x, y in minibatch_data:
            optimizer.zero_grad()  
            prediction = model(x)  
            loss = mse_loss(prediction, y)  
            loss.backward()  
            optimizer.step() 
        epochs.append(epoch)
        losses.append(loss.item())
        #Print loss every 100 epochs
        if (epoch + 1) % 100 == 0:
            print('Epoch: {} Loss {}'.format(epoch + 1, loss.item()))
    weight = model.weight
    w_opt = weight / max_powers
    return w_opt.reshape(M+1, 1)