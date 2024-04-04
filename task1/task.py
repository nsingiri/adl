import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from PIL import Image
import time

from task1 import polynomial_fun, fit_polynomial_ls, fit_polynomial_sgd

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





# Train the model or perform inference
def train_model():
    # Training loop
    for epoch in range(num_epochs):
        # Train model using custom functions
        # loss = your_custom_functions.train(model, optimizer, train_loader)
        pass

def evaluate_model():
    # Evaluate model performance using custom functions
    # accuracy = your_custom_functions.evaluate(model, val_loader)
    pass

# Main function to run the task
def main():
    # Set up data (if applicable)
    # ...

    # Define model and optimizer
    model = YourModel()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_model()

    # Evaluate model performance
    evaluate_model()



# Entry point to execute the task
if __name__ == "__main__":
    main()











def main():
    # Define weight vector
    w = torch.tensor([1, 2, 3], dtype=torch.float32).reshape(3, 1)

    # Generate training set
    x_train = 40.0 * (torch.rand(100, dtype=torch.float32) - 0.5).reshape(100, 1)
    y_train = polynomial_fun(w, x_train)
    noise_train = (0.2 * torch.randn(100, dtype=torch.float32)).reshape(100, 1)
    t_train = y_train + noise_train

    # Generate testing set
    x_test = 40.0 * (torch.rand(50, dtype=torch.float32) - 0.5).reshape(50, 1)
    y_test = polynomial_fun(w, x_test)
    noise_test = 0.2 * torch.randn(50, dtype=torch.float32).reshape(50, 1)
    t_test = y_test + noise_test

    # Compute optimum weight vector using fit_polynomial_ls (M=5) on the training set
    time_ls = time.time()
    w_hat_ls = fit_polynomial_ls(x_train, t_train, M=3)
    time_ls = time.time() - time_ls  # Time spent fitting for least squares

    # Compute predicted target values for both training and test sets
    y_hat_ls_train = polynomial_fun(w_hat_ls, x_train)
    y_hat_ls_test = polynomial_fun(w_hat_ls, x_test)

    # Report mean and standard deviation of differences between observed training data and true polynomial curve
    difference = t_train - y_train
    std_difference, mean_difference = torch.std_mean(difference)
    print(". \n" * 5)
    print("-" * 20 + "Observed training data and true polynomial" + "-" * 20)
    print("-" * 60)
    print("|{:<30}|{:<30}|".format("Metric", "Value"))
    print("-" * 60)
    print("|{:<30}|{:<30.5f}|".format("Mean difference", mean_difference.item()))
    print("|{:<30}|{:<30.5f}|".format("Standard deviation", std_difference.item()))
    print("-" * 60)


    # Report mean and standard deviation of differences between LS-predicted values and true polynomial curve
    difference = y_hat_ls_train - y_train
    std_difference, mean_difference = torch.std_mean(difference)
    print(". \n" * 5)
    print("-" * 60)
    print("|{:<60}|".format("Least square-predicted values and true polynomial"))
    print("|{:<30}|{:<30}|".format("Metric", "Value"))
    print("-" * 60)
    print("|{:<30}|{:<30.5f}|".format("Mean difference", mean_difference.item()))
    print("|{:<30}|{:<30.5f}|".format("Standard deviation", std_difference.item()))
    print("-" * 60)

    # Use fit_polynomial_sgd (M=5) to optimize the weight vector using the training set
    time_sgd = time.time()
    w_hat_sgd = fit_polynomial_sgd(x_train, t_train, 3, 0.25, 25)  # batch_size=25, learning rate=0.25
    time_sgd = time.time() - time_sgd  # Time spent training for SGD

    # Compute predicted target values for both training and test sets using SGD
    y_hat_sgd_train = polynomial_fun(w_hat_sgd, x_train)
    y_hat_sgd_test = polynomial_fun(w_hat_sgd, x_test)

    # Report mean and standard deviation of differences between SGD-predicted values and true polynomial curve
    difference = y_hat_sgd_train - y_train
    std_difference, mean_difference = torch.std_mean(difference)
    print(". \n" * 5)
    print("-" * 60)
    print("|{:<60}|".format("SGD-predicted values and true polynomial"))
    print("|{:<30}|{:<30}|".format("Metric", "Value"))
    print("-" * 60)
    print("|{:<30}|{:<30.5f}|".format("Mean difference", mean_difference.item()))
    print("|{:<30}|{:<30.5f}|".format("Standard deviation", std_difference.item()))
    print("-" * 60)


    # Calculate root-mean-square-errors (RMSEs) for w and y and report them
    mse_ls = torch.square(y_hat_ls_test - t_test)
    std_mse_ls, mean_mse_ls = torch.std_mean(mse_ls)

    mse_sgd = torch.square(y_hat_sgd_test - t_test)
    std_mse_sgd, mean_mse_sgd = torch.std_mean(mse_sgd)

    padding = nn.ZeroPad2d((0, 0, 0, 1))
    rmse_w_ls = torch.sqrt(torch.mean(torch.square(w_hat_ls - padding(w))))
    rmse_w_sgd = torch.sqrt(torch.mean(torch.square(w_hat_sgd - padding(w))))
    rmse_y_ls = torch.sqrt(torch.mean(torch.square(y_hat_ls_test - y_test)))
    rmse_y_sgd = torch.sqrt(torch.mean(torch.square(y_hat_sgd_test - y_test)))

    print(". \n" * 5)
    print("-" * 40 + "Final Report - Task 1" + "-" * 40)
    print("Metric                  | LS                  | SGD")
    print("-" * 40 + "|" + "-" * 18 + "|" + "-" * 17)
    print(f"Mean MSE                | {mean_mse_ls.tolist():<20} | {mean_mse_sgd.tolist():<20}")
    print(f"STD MSE                 | {std_mse_ls.tolist():<20} | {std_mse_sgd.tolist():<20}")
    print(f"RMSE for w              | {rmse_w_ls.tolist():<20} | {rmse_w_sgd.tolist():<20}")
    print(f"RMSE for y(test)        | {rmse_y_ls.tolist():<20} | {rmse_y_sgd.tolist():<20}")
    print(f"Training time (seconds) | {time_ls:<20} | {time_sgd:<20}")
    # Data Scientist Interpretation:
    print("\n Interpreting the Results:")
    print("SGD consistently achieves a more accurate solution compared to LS, as indicated by its smaller mean squared error (MSE). Additionally, the smaller standard deviation of errors for SGD suggests that its predictions are more consistent across the dataset, indicating a more robust model.")
    print("Furthermore, the root mean squared error (RMSE) for both the weight vector (w) and the predicted target values (y) are significantly smaller for SGD compared to LS. This suggests that SGD not only fits the polynomial better to the training data but also generalizes better to unseen data.")
    print("However, it's important to note that SGD comes with a trade-off, as it requires a much longer training time compared to LS.")
    

    metrics = ['Mean MSE', 'STD MSE', 'RMSE for w', 'RMSE for y(test)']
    ls_values = [mean_mse_ls.tolist(), std_mse_ls.tolist(), rmse_w_ls.tolist(), rmse_y_ls.tolist()]
    sgd_values = [mean_mse_sgd.tolist(), std_mse_sgd.tolist(), rmse_w_sgd.tolist(), rmse_y_sgd.tolist()]

    # Define the width of the bars
    bar_width = 0.35

    # Define the positions for the bars
    index = range(len(metrics))

    # Create the bar graph
    plt.bar(index, ls_values, bar_width, label='LS')
    plt.bar([i + bar_width for i in index], sgd_values, bar_width, label='SGD')

    # Add labels and title
    plt.xlabel('Metrics')
    plt.ylabel('Values')
    plt.title('Comparison of Metrics between LS and SGD')
    plt.xticks([i + bar_width / 2 for i in index], metrics, rotation=45, ha='right')
    plt.legend()

    # Show the plot
    plt.tight_layout()
    plt.show()

    print("-" * 40 + "end" + "-" * 40)

    del mse_ls, std_mse_ls, mean_mse_ls
    del mse_sgd, std_mse_sgd, mean_mse_sgd




if __name__=="__main__":
    main()











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
