# %% [markdown]
# # Locally Weighted Regression Model Implementation

# %% [markdown]
# This implementation of the Locally Weighted Regression algorithm is based on the following resources:
# - [VTU Pulse](https://vtupulse.com/machine-learning/locally-weighted-regression-algorithm-in-python/#:~:text=Implement%20the%20non-parametric%20Locally%20Weighted%20Regression%20Algorithm%20in)
# - [Wikipedia](https://en.wikipedia.org/wiki/Local_regression)
# 
# The algorithm is a non-parametric regression method that uses a weighted linear regression to predict the output of a new data point. The weights are calculated based on the distance between the new data point and the training data points. The closer the training data point is to the new data point, the higher the weight assigned to it.
# 
# The algorithm is defined by the following steps:
# 1. For each new data point, calculate the weights for each training data point based on the distance between them.
# 2. Fit a linear regression model using the training data points and the weights calculated in step 1.
# 3. Predict the output of the new data point using the linear regression model.
# 
# The kernel function used to calculate the weights is the Gaussian kernel function, which is defined as:
# $$
# w(i) = e^{-\frac{(x^{(i)} - x)^2}{2\tau^2}}
# $$
# where $x^{(i)}$ is the input feature of the training data point, $x$ is the input feature of the new data point, and $\tau$ is the bandwidth parameter that controls the width of the kernel.
# 
# The linear regression model is defined as:
# $$
# \theta = (X^TWX)^{-1}X^TWY
# $$
# where $X$ is the matrix of input features of the training data points, $W$ is the diagonal matrix of weights, and $Y$ is the vector of output values of the training data points.
# 
# A variation of the normal equation is used to calculate the weights in the linear regression model:
# $$
# \theta = (X^TWX +  I)^{-1}X^TWY
# $$
# where $I$ is the identity matrix.
# 
# The accuracy of the algorithm can be improved by tuning the bandwidth parameter $\tau$ and the mean squared error can be used as the evaluation metric.
# 
# The implementation of the Locally Weighted Regression algorithm is below, with the following functions:
# - `kernel(point, data, tau)`: Calculate the weights for each training data point based on the distance between the new data point and the training data points.
# - `local_weight(point, data, tau)`: Find the weights for each training data point using the `kernel` function, and a variation of the normal equation to calculate the weights in the linear regression model.
# - `local_weight_regression(X_b, Y, tau)`: Fit a linear regression model using the training data points and the weights calculated in the `local_weight` function.
# - `compute_mse(Y, Y_pred)`: Calculate the mean squared error between the predicted output and the actual output.

# %% [markdown]
# ## Implementation

# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# %% [markdown]
# The kernel function used in this implementation is the Gaussian kernel function, but other kernel functions can be used as well. The bandwidth parameter $\tau$ can be tuned to improve the accuracy of the algorithm. The kernel function is used to calculate the weights for each training data point based on the distance between the new data point and the training data points. The weights are then used in the linear regression model to predict the output of the new data point.

# %%
def kernel(point, xmat, k):
    """Calculates the kernel function for a given point and data set X_b.
    This kernel function is a Gaussian kernel which assigns weights to each
    point in the data set based on the distance between the point and the
    other points in the data set (based on a Gaussian distribution).
    Args:
        point: the point to calculate the kernel function for
        xmat: the data set
        k: the kernel width (also called the bandwidth, and represented by tau)
    Returns:
        weights: the weights for each point in the data set
    """
    m, n = np.shape(xmat)
    weights = np.mat(np.eye(m))
    for j in range(m):
        diff = point - xmat[j]
        weights[j, j] = np.exp(float(diff * diff.T) / (-2.0 * k**2))  
    return weights

# %% [markdown]
# The `local_weight` function calculates the weights for each training data point using the `kernel` function and a variation of the normal equation to calculate the weights in the linear regression model. The weights are then used in the linear regression model to predict the output of the new data point. To avoid singular matrix errors or linearly dependent features, the pseudo-inverse method is used to calculate the weights in the linear regression model.

# %%
def local_weight(point, xmat, ymat, k):
    """Calculates the local weight for a given point.
    This function calculates the local weight for a given point by using the
    kernel function to assign weights to each point in the data set, and then
    solving the weighted normal equation to find the local regression coefficients.
    Args:
        point: the point to calculate the local weight for
        xmat: the data set
        ymat: the target values
        k: the kernel width (also called the bandwidth, and represented by tau)
    Returns:
        W: the local weight for the given point
    """
    wei = kernel(point, xmat, k)
    XTWX = xmat.T * (wei * xmat)
    XTWY = xmat.T * (wei * ymat)
    W = np.linalg.pinv(XTWX) * XTWY  
    return W

# %% [markdown]
# The `local_weight_regression` function fits a linear regression model using the training data points and the weights calculated in the `local_weight` function. The linear regression model is used to predict the output of the new data point.

# %%
def local_weight_regression(xmat, ymat, k):
    """Calculates the local weight regression for a given data set.
    This function calculates the local weight regression for a given data set
    by iterating through each point in the data set, calculating the local weight
    for each point, and then using the local weight to make a prediction for each point.
    Args:
        xmat: the data set
        ymat: the target values
        k: the kernel width (also called the bandwidth, and represented by tau)
    Returns:
        Y_pred: the predicted values
    """
    m, n = np.shape(xmat)
    ypred = np.zeros(m)
    for i in range(m):
        W = local_weight(xmat[i], xmat, ymat, k)
        ypred[i] = float(np.dot(xmat[i], W))  
    return ypred

# %% [markdown]
# `Mean Squared Error` is used as the evaluation metric to measure the accuracy of the algorithm. The mean squared error is calculated as the average of the squared differences between the predicted output and the actual output.

# %%
def compute_mse(Y_true, Y_pred):
    """Computes the Mean Squared Error (MSE) of the model.
    This function computes the Mean Squared Error (MSE) of the model by
    calculating the average squared difference between the actual target values
    and the predicted values.
    Args:
        Y_true: the actual target values
        Y_pred: the predicted values
    Returns:
        mse: Mean Squared Error
    """
    mse = np.mean((Y_true - Y_pred) ** 2)
    return mse

# %% [markdown]
# The function below seeks to find the optimal bandwidth parameter $\tau$ that minimizes the mean squared error of the algorithm. he steps are as follows:
# 1. Generate a range of values for the bandwidth parameter $\tau$ from 0.1 to 1.0, in increments of 0.001.
# 2. For each value of $\tau$, fit the Locally Weighted Regression model using the training data points and calculate the mean squared error. Store the mean squared error for each value of $\tau$.
# 3. Find the value of $\tau$ that minimizes the mean squared error and use it to fit the final Locally Weighted Regression model.
# 4. Plot the mean squared error for each value of $\tau` to visualize the relationship between the bandwidth parameter and the mean squared error.

# %%
def find_optimal_tau(X_b, y_mat):
    """Finds the optimal tau for the local weight regression model.
    This function finds the optimal tau for the local weight regression model
    by iterating through a range of tau values, calculating the Mean Squared Error (MSE)
    for each tau, and then selecting the tau value that minimizes the MSE. It then
    plots the MSE vs Tau graph to visualize the relationship between the two.
    Args:
        X_b: the data set
        y_mat: the target values
    Returns:
        optimal_tau: the optimal tau value
    """
    taus = np.linspace(0.1, 1, 20)
    mses = []
    for tau in taus:
        y_pred = local_weight_regression(X_b, y_mat, tau)
        mse = compute_mse(y_mat, y_pred)
        mses.append(mse)
    
    optimal_tau = taus[np.argmin(mses)]
    
    plt.plot(taus, mses, 'b-', label='MSE')
    plt.plot(taus[np.argmin(mses)], min(mses), 'ro', label='Minimum MSE/Optimal Tau')
    plt.xlabel('Tau')
    plt.ylabel('MSE')
    plt.legend()
    plt.title('Tau vs MSE')
    plt.show()
    
    return optimal_tau

# %%
def plot_regression_line(X, y, predictions, feature_name, target_name, tau = 0):
    sorted_indices = np.argsort(X)
    X_sorted = X[sorted_indices]
    predictions_sorted = predictions[sorted_indices]

    plt.scatter(X, y, color='blue', alpha=0.5, label='Original Data')
    plt.plot(X_sorted, predictions_sorted, color='red', label='Regression Line')
    plt.xlabel(feature_name)
    plt.ylabel(target_name)
    plt.title(f'{target_name} vs {feature_name}')
    plt.legend()
    plt.show()

# %%
def effect_of_tau_on_regression_line(X_b, y_mat, X, y, y_mean, y_std):
    """Plots the effect of tau on the regression line.
    This function plots the effect of tau on the regression line by
    plotting the regression line for different tau values.
    Args:
        X_b: the data set
        y_mat: the target values
        X: the original feature values
        y: the original target values
        y_mean: the mean of the original target values
        y_std: the standard deviation of the original target values
    """
    taus = np.linspace(0.01, 1, 5)
    plt.figure(figsize=(15, 10))
    for i, tau in enumerate(taus):
        plt.subplot(2, 3, i + 1)
        predictions_normalized = local_weight_regression(X_b, y_mat, tau)
        predictions = predictions_normalized * y_std + y_mean
        plot_regression_line(X, y, predictions, 'Size of Plot (sq. meters)', 'Land Price (GHS)', tau)
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## Testing the Implementation

# %% [markdown]
# The model is tested on a synthetic dataset, provided in-class when we looked at the Locally Weighted Regression algorithm. The model is only trained on the size of the land and the price of the house, and the goal is to predict the price of a house given the size of the land.

# %%
data = pd.read_csv('simulated_land_price_data.csv')

# %%
data.info()

# %%
data.head()

# %%
plt.scatter(data['Size of Plot (sq. meters)'], data['Land Price (GHS)'], color='blue', alpha=0.5, label='Original Data')
plt.xlabel('Size of Plot (sq. meters)')
plt.ylabel('Land Price (GHS)')
plt.title('Land Price Data Distribution')
plt.legend()

# %% [markdown]
# Extracting the feature of interest (size of the land) and the target variable (price of the house) from the dataset:

# %%
X = np.array(data['Size of Plot (sq. meters)'])
y = np.array(data['Land Price (GHS)'])

# %%
X

# %%
y

# %% [markdown]
# The target values are normalized to improve the performance of the algorithm, as I observed that the algorithm performs better when the target values displayed less variance. As such, predictions are made on the normalized target values, and the predictions are then denormalized to get the actual price of the house.

# %%
X_mean = np.mean(X)
X_std = np.std(X)
X_normalized = (X - X_mean) / X_std

y_mean = np.mean(y)
y_std = np.std(y)
y_normalized = (y - y_mean) / y_std

# %%
X_mat = np.mat(X_normalized).T
y_mat = np.mat(y_normalized).T

# %%
X_mat

# %%
y_mat

# %% [markdown]
# The intercept term is added to the feature matrix to account for the bias term in the linear regression model.

# %%
m = np.shape(X_mat)[0]
one = np.mat(np.ones(m)).T
X_b = np.hstack((one, X_mat))

# %%
X_b

# %%
optimal_tau = find_optimal_tau(X_b, y_mat)

# %%
print(f'Optimal Tau: {optimal_tau}')

# %% [markdown]
# Observations:
# The relationship betweeen the bandwidth parameter $\tau$ and the mean squared error (MSE) is used as a proxy to understand the effect of $\tau$ on the model's performance and prediction smoothness. The curve exhibits a steep decline in MSE for small values of $\tau$, followed by a gradual flattening as $\tau$ increases. The red dot at the leftmost part of the curve indicates the minimum MSE, which occurs when $\tau$ is very close to zero. As $\tau$ increases beyond this point, the MSE rapidly increases, and eventually, the curve plateaus. The plateau indicates that further increases in $\tau$ no longer significantly change the MSE, but the model is likely becoming too smooth.
# 
# Analysis of $\tau's$ Effect on Prediction Smoothness:
# - When \(\tau\) is small, the model prioritizes points very close to the data point being predicted, meaning the model heavily weights nearby points. This creates a regression line that is highly sensitive to local variations. As a result, the predictions tend to be less smooth, as the MSE is minimized implying it is fitting quite well to the data points. This explains why, for small \(\tau\), the MSE is minimized (the model fits very closely to training data), but it may perform poorly on new data (leading to increased MSE after \(\tau\) surpasses the optimal point).
#   
# - As \(\tau\) increases, the bandwidth expands, and the model starts to incorporate more distant points in the prediction, averaging out local fluctuations. This leads to a smoother regression line, as MSE is not minimized so the curve strays away frm individual data points, but at the cost of losing some of the finer details captured by smaller \(\tau\). At very large \(\tau\), the regression line becomes overly smooth, potentially leading to underfitting, where the model is too simplistic and fails to capture important trends in the data. This is evident from the flatter portion of the curve at larger \(\tau\), where the MSE stops decreasing and stabilizes at a higher value.
# 
# Explanation of \(\tau\)'s Influence on the Regression Line:
# - Small \(\tau\): The model becomes highly flexible, leading to a regression line that follows the training data closely. This results in high sensitivity to noise and sharp fluctuations, reducing the smoothness of the predictions. The risk is overfitting, where the model does not generalize well to new data.
#   
# - Large \(\tau\): The model becomes smoother as it gives equal weight to a broader range of points, leading to a more general prediction line that ignores local variations. While this reduces variance, it can lead to underfitting if important patterns are smoothed out too much.

# %% [markdown]
# ---

# %% [markdown]
# After the optimal bandwidth parameter $\tau$ is found, the Locally Weighted Regression model is fitted using the training data points and the optimal $\tau$. The model is then used to predict the output of the new data point. However, the predictions are made on the normalized target values, so the predictions are then denormalized to get the actual price of the house.

# %%
predictions_normalized = local_weight_regression(X_b, y_mat, optimal_tau)

# %% [markdown]
# Denormalizing the predicted output to get the actual price of the land:

# %%
predictions = predictions_normalized * y_std + y_mean

# %% [markdown]
# Printing each prediction and the actual price of the house to compare the results.

# %%
for i in range(m):
    print('predicted:', predictions[i], 'actual:', y[i])

# %% [markdown]
# The mean squared error of the final Locally Weighted Regression model is calculated to evaluate the accuracy of the algorithm.

# %%
error = compute_mse(y, predictions)
print(f'Mean Squared Error: {error:.8f}')

# %% [markdown]
# The original data is plotted along with the regression line to visualize the relationship between the size of the land and the price of the house. The regression line fits the data well, indicating that the Locally Weighted Regression model is accurate in predicting the price of a house given the size of the land.

# %%
plot_regression_line(X, y, predictions, 'Size of Plot (sq. meters)', 'Land Price (GHS)')

# %%
def predict_unseen_data(unseen_data, X_b, y_mat, optimal_tau, y_mean, y_std, X_mean, X_std):
    unseen_data_normalized = (unseen_data - X_mean) / X_std
    unseen_data_mat = np.mat(unseen_data_normalized).T
    unseen_data_b = np.hstack((np.ones((unseen_data_mat.shape[0], 1)), unseen_data_mat))

    unseen_predictions_normalized = np.zeros(unseen_data_b.shape[0])
    for i in range(unseen_data_b.shape[0]):
        W = local_weight(unseen_data_b[i], X_b, y_mat, optimal_tau)
        unseen_predictions_normalized[i] = float(np.dot(unseen_data_b[i], W))

    unseen_predictions = unseen_predictions_normalized * y_std + y_mean
    return unseen_predictions

unseen_data = np.array([287, 500, 120, 150, 160, 178])
unseen_predictions = predict_unseen_data(unseen_data, X_b, y_mat, optimal_tau, y_mean, y_std, X_mean, X_std)

for i, size in enumerate(unseen_data):
    print(f'Predicted price for plot size {size} sq. meters: {unseen_predictions[i]:.2f} GHS')

# %% [markdown]
# ## References
# https://vtupulse.com/machine-learning/locally-weighted-regression-algorithm-in-python/#:~:text=Implement%20the%20non-parametric%20Locally%20Weighted%20Regression%20Algorithm%20in


