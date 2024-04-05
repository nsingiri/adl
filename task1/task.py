import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from PIL import Image
import time

from task1 import polynomial_fun, fit_polynomial_ls, fit_polynomial_sgd


def compute_rmse(true_values, predicted_values):
    """
    Computes the root-mean-square-error.
    Args:
        true_values (torch.Tensor): 
        predicted_values (torch.Tensor): 
    Returns:
        y (torch.Tensor): root-mean-square-error value between true and predicted values
    """
    rmse = torch.sqrt(torch.mean((true_values - predicted_values) ** 2))
    return rmse


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

    # Compute optimum weight vector using fit_polynomial_ls for M=2,3,4 on the training set
    time_ls_two = time.time()
    w_hat_ls_two = fit_polynomial_ls(x_train, t_train, M=2)
    time_ls_two = time.time() - time_ls_two
    time_ls_three = time.time()
    w_hat_ls_three = fit_polynomial_ls(x_train, t_train, M=3)
    time_ls_three = time.time() - time_ls_three
    time_ls_four = time.time()
    w_hat_ls_four = fit_polynomial_ls(x_train, t_train, M=4)
    time_ls_four = time.time() - time_ls_four

    # Compute predicted target values for both training and test sets
    # M = 2
    y_hat_ls_train_two = polynomial_fun(w_hat_ls_two, x_train)
    y_hat_ls_test_two = polynomial_fun(w_hat_ls_two, x_test)
    # M = 3
    y_hat_ls_train_three = polynomial_fun(w_hat_ls_three, x_train)
    y_hat_ls_test_three = polynomial_fun(w_hat_ls_three, x_test)
    # M = 4
    y_hat_ls_train_four = polynomial_fun(w_hat_ls_four, x_train)
    y_hat_ls_test_four = polynomial_fun(w_hat_ls_four, x_test)

    # Difference between observed training data and the true polynomial curve
    difference = t_train - y_train
    mean_diff = torch.mean(difference)
    std_diff = torch.std(difference)

    print('Mean of difference (between observed training data and true polynomial curve) : {}'.format(mean_diff))
    print('Standard Deviation of difference (between observed training data and true polynomial curve) : {}'.format(std_diff))

    # Difference between LS-predicted values and the true polynomial curve
    # M = 2
    ls_difference_two = y_hat_ls_train_two - y_train
    mean_diff_ls_two = torch.mean(ls_difference_two)
    std_diff_ls_two = torch.std(ls_difference_two)
    print('M=2: Mean of difference (between LS-predicted values and true polynomial curve) : {}'.format(mean_diff_ls_two))
    print('M=2: Standard Deviation of difference (between LS-predicted values and true polynomial curve) : {}'.format(std_diff_ls_two))
    # M = 3
    ls_difference_three = y_hat_ls_train_three - y_train
    mean_diff_ls_three = torch.mean(ls_difference_three)
    std_diff_ls_three = torch.std(ls_difference_three)
    print('M=3: Mean of difference (between LS-predicted values and true polynomial curve) : {}'.format(mean_diff_ls_three))
    print('M=3: Standard Deviation of difference (between LS-predicted values and true polynomial curve) : {}'.format(std_diff_ls_three))
    # M = 4
    ls_difference_four = y_hat_ls_train_four - y_train
    mean_diff_ls_four = torch.mean(ls_difference_four)
    std_diff_ls_four = torch.std(ls_difference_four)
    print('M=4: Mean of difference (between LS-predicted values and true polynomial curve) : {}'.format(mean_diff_ls_four))
    print('M=4: Standard Deviation of difference (between LS-predicted values and true polynomial curve) : {}'.format(std_diff_ls_four))
        
    # Compute optimum weight vector using fit_polynomial_sgd for M=2,3,4 on the training set
    print("M=2:")
    time_sgd_two = time.time()
    w_hat_sgd_two = fit_polynomial_sgd(x_train, t_train, M=2, learning_rate=0.1, minibatch_size=10) 
    time_sgd_two = time.time() - time_sgd_two
    print("M=3:")
    time_sgd_three = time.time()
    w_hat_sgd_three = fit_polynomial_sgd(x_train, t_train, M=3, learning_rate=0.1, minibatch_size=10)
    time_sgd_three = time.time() - time_sgd_three
    print("M=4:")
    time_sgd_four = time.time()
    w_hat_sgd_four = fit_polynomial_sgd(x_train, t_train, M=4, learning_rate=0.1, minibatch_size=10)
    time_sgd_four = time.time() - time_sgd_four

    # Compute predicted target values for both training and test sets using fit_polynomial_sgd
    # M = 2
    y_hat_sgd_train_two = polynomial_fun(w_hat_sgd_two, x_train)
    y_hat_sgd_test_two = polynomial_fun(w_hat_sgd_two, x_test)
    # M = 3
    y_hat_sgd_train_three = polynomial_fun(w_hat_sgd_three, x_train)
    y_hat_sgd_test_three = polynomial_fun(w_hat_sgd_three, x_test)
    # M = 4
    y_hat_sgd_train_four = polynomial_fun(w_hat_sgd_four, x_train)
    y_hat_sgd_test_four = polynomial_fun(w_hat_sgd_four, x_test)

    # Difference between SGD-predicted values and the true polynomial curve
    # M = 2
    sgd_difference_two = y_hat_sgd_train_two - y_train
    mean_diff_sgd_two = torch.mean(sgd_difference_two)
    std_diff_sgd_two = torch.std(sgd_difference_two)
    print('M=2: Mean of difference (between LS-predicted values and true polynomial curve) : {}'.format(mean_diff_sgd_two))
    print('M=2: Standard Deviation of difference (between LS-predicted values and true polynomial curve) : {}'.format(std_diff_sgd_two))
    # M = 3
    sgd_difference_three = y_hat_sgd_train_three - y_train
    mean_diff_sgd_three = torch.mean(sgd_difference_three)
    std_diff_sgd_three = torch.std(sgd_difference_three)
    print('M=3: Mean of difference (between LS-predicted values and true polynomial curve) : {}'.format(mean_diff_sgd_three))
    print('M=3: Standard Deviation of difference (between LS-predicted values and true polynomial curve) : {}'.format(std_diff_sgd_three))
    # M = 4
    sgd_difference_four = y_hat_sgd_train_four - y_train
    mean_diff_sgd_four = torch.mean(sgd_difference_four)
    std_diff_sgd_four = torch.std(sgd_difference_four)
    print('M=4: Mean of difference (between LS-predicted values and true polynomial curve) : {}'.format(mean_diff_sgd_four))
    print('M=4: Standard Deviation of difference (between LS-predicted values and true polynomial curve) : {}'.format(std_diff_sgd_four))


    # M = 2:
    rmse_y_ls_two = compute_rmse(y_hat_ls_test_two, y_test)
    rmse_y_sgd_two = compute_rmse(y_hat_sgd_test_two, y_test)
    rmse_w_ls_two = compute_rmse(w_hat_ls_two, w)
    rmse_w_sgd_two = compute_rmse(w_hat_sgd_two, w)


    print("M=2: RMSE of y using LS is {}".format(rmse_y_ls_two))
    print("M=2: RMSE of y using SGD is {}".format(rmse_y_sgd_two))
    print("M=2: RMSE of w using LS is {}".format(rmse_w_ls_two))
    print("M=2: RMSE of w using SGD is {}".format(rmse_w_sgd_two))


    w_padded_three = nn.functional.pad(w, (0, 0, 0, 1), mode='constant', value=0)
    # M = 3: 
    rmse_y_ls_three = compute_rmse(y_hat_ls_test_three, y_test)
    rmse_y_sgd_three = compute_rmse(y_hat_sgd_test_three, y_test)
    rmse_w_ls_three = compute_rmse(w_hat_ls_three, w_padded_three)
    rmse_w_sgd_three = compute_rmse(w_hat_sgd_three, w_padded_three)


    print("M=3: RMSE of y using LS is {}".format(rmse_y_ls_three))
    print("M=3: RMSE of y using SGD is {}".format(rmse_y_sgd_three))
    print("M=3: RMSE of w using LS is {}".format(rmse_w_ls_three))
    print("M=3: RMSE of w using SGD is {}".format(rmse_w_sgd_three))

    w_padded_four = nn.functional.pad(w, (0, 0, 0, 2), mode='constant', value=0)
    # M = 4: 
    rmse_y_ls_four = compute_rmse(y_hat_ls_test_four, y_test)
    rmse_y_sgd_four = compute_rmse(y_hat_sgd_test_four, y_test)
    rmse_w_ls_four = compute_rmse(w_hat_ls_four, w_padded_four)
    rmse_w_sgd_four = compute_rmse(w_hat_sgd_four, w_padded_four)


    print("M=4: RMSE of y using LS is {}".format(rmse_y_ls_four))
    print("M=4: RMSE of y using SGD is {}".format(rmse_y_sgd_four))
    print("M=4: RMSE of w using LS is {}".format(rmse_w_ls_four))
    print("M=4: RMSE of w using SGD is {}".format(rmse_w_sgd_four))

    print("M = 2: Time spent training ls: {}".format(time_ls_two))
    print("M = 2: Time spent training sgd: {}".format(time_sgd_two))
    print("M = 3: Time spent training ls: {}".format(time_ls_three))
    print("M = 3: Time spent training sgd: {}".format(time_sgd_three))
    print("M = 4: Time spent training ls: {}".format(time_ls_four))
    print("M = 4: Time spent training sgd: {}".format(time_sgd_four))


if __name__=="__main__":
    main()