from task1 import polynomial_fun, fit_polynomial_ls, fit_polynomial_sgd
import torch
import numpy as np
import time

#generate training and test set
train = np.random.uniform(-20.0, 20.0, 20)
test = np.random.uniform(-20.0, 20.0, 10)
w = [1, 2, 3]
y_train = [polynomial_fun(w, x) for x in train]
y_test = [polynomial_fun(w, x) for x in test]
t_train = [x + np.random.normal(scale=0.5) for x in y_train]
t_test = [x + np.random.normal(scale=0.5) for x in y_test]

#compute optimum weight vector
w_hat = fit_polynomial_ls(train, t_train, 2) #M=2
y_hat_train = polynomial_fun(w_hat, train)
y_hat_test = polynomial_fun(w_hat, test)















# def generate_data(M, num_train, num_test):
#
#     # Generate training data
#     x_train = np.random.uniform(-20.0, 20.0, 20)
#     x_test = np.random.uniform(-20.0, 20.0, 10)
#     w = torch.tensor([1.0, 2.0, 3.0])
#
#     y_train_true = polynomial_fun(x_train, torch.tensor([1.0, 2.0, 3.0]))  # True polynomial curve
#     t_train = y_train_true + torch.randn_like(y_train_true) * 0.5  # Add Gaussian noise
#
#     # Generate test data
#     y_test_true = polynomial_fun(x_test, torch.tensor([1.0, 2.0, 3.0]))  # True polynomial curve
#     t_test = y_test_true + torch.randn_like(y_test_true) * 0.5  # Add Gaussian noise
#
#     return x_train, t_train, x_test, t_test
#
#
# def compute_rmse(true_values, predicted_values):
#     return torch.sqrt(torch.mean((true_values - predicted_values) ** 2))
#
#
# def main():
#     Ms = [2, 3, 4]  # Polynomial degrees
#     num_train = 20
#     num_test = 10
#
#     for M in Ms:
#         print(f"Degree of polynomial: {M}")
#
#         # Generate data
#         x_train, t_train, x_test, t_test = generate_data(M, num_train, num_test)
#
#         # Least square fitting
#         start_time_ls = time.time()
#         w_ls = fit_polynomial_ls(x_train, t_train, M)
#         y_train_ls = polynomial_fun(x_train, w_ls)
#         y_test_ls = polynomial_fun(x_test, w_ls)
#         train_rmse_w_ls = compute_rmse(torch.tensor([1.0, 2.0, 3.0]), w_ls)
#         train_rmse_y_ls = compute_rmse(t_train, y_train_ls)
#         test_rmse_y_ls = compute_rmse(t_test, y_test_ls)
#         end_time_ls = time.time()
#
#         print(f"Least Squares - Time spent: {end_time_ls - start_time_ls:.4f}s")
#         print(
#             f"Train RMSE (w): {train_rmse_w_ls:.4f}, Train RMSE (y): {train_rmse_y_ls:.4f}, Test RMSE (y): {test_rmse_y_ls:.4f}")
#
#         # Stochastic gradient descent
#         start_time_sgd = time.time()
#         w_sgd = fit_polynomial_sgd(x_train, t_train, M, learning_rate=0.01, minibatch_size=10, num_epochs=1000)
#         y_train_sgd = polynomial_fun(x_train, w_sgd)
#         y_test_sgd = polynomial_fun(x_test, w_sgd)
#         train_rmse_w_sgd = compute_rmse(torch.tensor([1.0, 2.0, 3.0]), w_sgd)
#         train_rmse_y_sgd = compute_rmse(t_train, y_train_sgd)
#         test_rmse_y_sgd = compute_rmse(t_test, y_test_sgd)
#         end_time_sgd = time.time()
#
#         print(f"Stochastic Gradient Descent - Time spent: {end_time_sgd - start_time_sgd:.4f}s")
#         print(
#             f"Train RMSE (w): {train_rmse_w_sgd:.4f}, Train RMSE (y): {train_rmse_y_sgd:.4f}, Test RMSE (y): {test_rmse_y_sgd:.4f}")
#         print()
#
#
# if __name__ == "__main__":
#     main()
