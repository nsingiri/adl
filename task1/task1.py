#conda activate comp0197-cw1-pt
import torch
import numpy as np
from PIL import Image


def polynomial_fun(w, x):
    """
    Evaluates a polynomial function given the weight vector w and an input scalar variable x.
    Args:
        w (list): Weight vector of size M + 1.
        x (int or float): Input scalar variable.
    Returns:
        y (int or float): Output value of the polynomial function.
    """
    M = len(w)
    powers = np.arange(M)
    y = np.sum(w * np.power(x, powers)) #iterate from m=0 to m=M
    return y

# Define polynomial function - PyTorch Version
# def polynomial_fun(w, x):
#     """
#     Evaluates a polynomial function given the weight vector w and an input scalar variable x.
#     Args:
#         w (torch.Tensor): Weight vector of size M + 1.
#         x (int or float): Input scalar variable.
#     Returns:
#         y (int or float): Output value of the polynomial function.
#     """
#    M = len(w)
#    return torch.sum(w * (x.unsqueeze(1) ** torch.arange(M, dtype=torch.float32)), dim=1)


def fit_polynomial_ls(x, t, M):





# def fit_polynomial_ls(x, t, M):
#     """
#     Fits a polynomial function using a least square solver.
#     Args:
#         x (torch.Tensor): Input data points of shape (N,).
#         t (torch.Tensor): Target values of shape (N,).
#         M (int): Polynomial degree.
#     Returns:
#         w_opt (torch.Tensor): Optimum weight vector of shape (M+1,).
#     """
#     # # Generate the Vandermonde matrix
#     # X = torch.pow(x.unsqueeze(1), torch.arange(M + 1, dtype=torch.float32))
#     # # Compute the pseudo-inverse of X
#     # X_pseudo_inv = torch.linalg.pinv(X)
#     #
#     # # Compute the optimal weight vector
#     # w_hat = torch.matmul(X_pseudo_inv, t)
#
#     # return w_hat
#
#     X_matrix = torch.tensor(x ** torch.arange(M + 1), dtype=torch.float32)
#     w_opt, _ = torch.linalg.lstsq(t.unsqueeze(1), X_matrix)
#     return w_opt.squeeze().numpy()
#     # w_opt = torch.linalg.lstsq(t.unsqueeze(1), X_matrix)
#     # return w_opt
#
#     sol = torch.linalg.lstsq(X_matrix, )

# ||AX - B||
# x = A-1B
# X_sol = func(A, B)
# linear regression soln --> (XTX)-1XTy
# minimize ||t - y||^2

def fit_polynomial_sgd(x, t, M, learning_rate, minibatch_size):
    """
    Fits a polynomial function using stochastic minibatch gradient descent.

    Args:
        x (torch.Tensor): Input data points of shape (N,).
        t (torch.Tensor): Target values of shape (N,).
        M (int): Polynomial degree.
        learning_rate (float): Learning rate for gradient descent.
        minibatch_size (int): Size of the minibatch.

    Returns:
        w_hat (torch.Tensor): Optimum weight vector of shape (M+1,).
    """
    num_epochs = 100
    # Initialize weights randomly
    w = torch.randn(M + 1, requires_grad=True)
    model = polynomial_fun(w, x)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, momentum=0.9)

    # Main training loop
    for epoch in range(num_epochs):

        optimizer.zero_grad()
        pred = model.forward(xTr)  # compute model predictions
        loss = mse_loss(pred, yTr) + reg_param * torch.norm(model.w)
        loss.backward()  # compute the gradient wrt loss
        optimizer.step()  # performs a step of gradient descent
        if (epoch + 1) % print_freq == 0:
            print('epoch {} loss {}'.format(epoch + 1, loss.item()))

        # Shuffle data
        indices = torch.randperm(x.size(0))
        x_shuffled = x[indices]
        t_shuffled = t[indices]
        # Mini-batch gradient descent
        #range = start, stop, step
        for i in range(0, x.size(0), minibatch_size):
            x_mb = x_shuffled[i:i + minibatch_size]
            t_mb = t_shuffled[i:i + minibatch_size]
            # Forward pass
            x_mb = x_mb.numpy()
            w = w.numpy()
            y_mb = polynomial_fun(x_mb, w)
            loss = torch.mean((y_mb - t_mb) ** 2)
            # Backward pass
            loss.backward()
            # Update weights
            with torch.no_grad():
                w -= learning_rate * w.grad
            # Zero gradients
            w.grad.zero_()

        # Print loss periodically
        if epoch % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}")

    return w.detach()



        optimizer.zero_grad()
        pred = model.forward(xTr)  # compute model predictions
        loss = mse_loss(pred, yTr) + reg_param * torch.norm(model.w)
        loss.backward()  # compute the gradient wrt loss
        optimizer.step()  # performs a step of gradient descent
        if (epoch + 1) % print_freq == 0:
            print('epoch {} loss {}'.format(epoch + 1, loss.item()))

    return model  # return trained model
