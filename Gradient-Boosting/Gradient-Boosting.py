import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import torch
from torch.distributions.normal import Normal
from torch.autograd import Variable
from typing import List, Optional


class GaussianGradientBoosting:
    def __init__(self,
                 learning_rate: float = 0.025,
                 max_depth: int = 1,
                 n_estimators: int = 100):
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.init_mu = None
        self.mu_trees = []
        self.init_sigma = None
        self.sigma_trees = []
        self.is_trained = False

    def predict(self, X: np.array) -> np.array:
        assert self.is_trained, "Model not trained yet!"
        mus = self._predict_mus(X).reshape(-1, 1)
        sigmas = np.exp(self._predict_log_sigmas(X)).reshape(-1, 1)
        return np.concatenate([mus, sigmas], axis=1)

    def predict_mean(self, X: np.array) -> np.array:
        assert self.is_trained, "Model not trained yet!"
        return self._predict_mus(X)

    def _predict_mus(self, X: np.array) -> np.array:
        output = np.full(len(X), self.init_mu)
        for tree in self.mu_trees:
            output += self.learning_rate * tree.predict(X)
        return output

    def _predict_log_sigmas(self, X: np.array) -> np.array:
        output = np.full(len(X), self.init_sigma)
        for tree in self.sigma_trees:
            output += self.learning_rate * tree.predict(X)
        return output

    def fit(self, X: np.array, y: np.array) -> None:
        self._fit_initial(y)
        
        for i in range(self.n_estimators):
            y_pred = self._predict_raw(X)
            gradients = self._get_gradients(y, y_pred)
            
            mu_tree = DecisionTreeRegressor(max_depth=self.max_depth, random_state=i)
            mu_tree.fit(X, gradients[:, 0])
            self.mu_trees.append(mu_tree)
            
            sigma_tree = DecisionTreeRegressor(max_depth=self.max_depth, random_state=i)
            sigma_tree.fit(X, gradients[:, 1])
            self.sigma_trees.append(sigma_tree)
            
            if (i + 1) % 20 == 0:
                current_mu = self._predict_mus(X)
                mse = mean_squared_error(y, current_mu)
                print(f"Iteration {i+1}/{self.n_estimators} - MSE: {mse:.4f}")
        
        self.is_trained = True

    def _fit_initial(self, y: np.array) -> None:
        assert not self.is_trained, "Model already trained!"
        self.init_mu = np.mean(y)
        self.init_sigma = np.log(np.std(y) + 1e-6)

    def _get_gradients(self, y: np.array, y_pred: np.array) -> np.array:
        y_torch = torch.tensor(y, dtype=torch.float32)
        y_pred_torch = Variable(torch.tensor(y_pred, dtype=torch.float32), requires_grad=True)
        
        normal_dist = Normal(y_pred_torch[:, 0], torch.exp(y_pred_torch[:, 1]))
        log_prob = normal_dist.log_prob(y_torch).sum()
        log_prob.backward()
        
        return y_pred_torch.grad.numpy()

    def _predict_raw(self, X: np.array) -> np.array:
        mus = self._predict_mus(X).reshape(-1, 1)
        log_sigmas = self._predict_log_sigmas(X).reshape(-1, 1)
        return np.concatenate([mus, log_sigmas], axis=1)


def load_data():
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, housing.feature_names


def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, r2


def plot_results(y_test, y_pred, y_std):
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Actual vs Predicted House Prices')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.scatter(y_pred, y_test - y_pred, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Price')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("=== Loading California Housing Dataset ===")
    X_train, X_test, y_train, y_test, feature_names = load_data()
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"Features: {feature_names}")
    
    print("\n=== Training Gaussian Gradient Boosting ===")
    model = GaussianGradientBoosting(
        learning_rate=0.05,
        max_depth=3,
        n_estimators=100
    )
    model.fit(X_train, y_train)
    
    print("\n=== Evaluating Model ===")
    y_pred_mean = model.predict_mean(X_test)
    y_pred = model.predict(X_test)
    y_pred_std = y_pred[:, 1]
    
    mse, rmse, r2 = evaluate_model(y_test, y_pred_mean)
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    print("\n=== Visualizing Results ===")
    plot_results(y_test, y_pred_mean, y_pred_std)