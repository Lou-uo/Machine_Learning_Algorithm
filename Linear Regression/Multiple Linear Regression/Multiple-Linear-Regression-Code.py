# Step 1: Data Preprocessing
## Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

## Importing the dataset
csv_path = Path(__file__).resolve().parent / '50_Startups.csv'
dataset = pd.read_csv(str(csv_path))
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 4].values

## Encoding Categorical data
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer([('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = ct.fit_transform(X)

## Avoiding the Dummy Variable Trap
X = X[:, 1:]

## Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Step 2: Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Step 3: Predicting the Test set results
y_pred = regressor.predict(X_test)

# Step 4: Visualization
## Visualize actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(Y_test, y_pred, color='red')
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'b--')
plt.title('Actual vs Predicted Profit')
plt.xlabel('Actual Profit')
plt.ylabel('Predicted Profit')
plt.grid(True)
plt.show()

## Feature importance (coefficients)
feature_names = ['State_2', 'State_3', 'R&D Spend', 'Administration', 'Marketing Spend']
coefficients = regressor.coef_

plt.figure(figsize=(10, 6))
plt.barh(feature_names, coefficients, color='skyblue')
plt.title('Feature Coefficients (Impact on Profit)')
plt.xlabel('Coefficient Value')
plt.grid(True)
plt.show()

## Residual plot
residuals = Y_test - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, color='green')
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residuals vs Predicted Values')
plt.xlabel('Predicted Profit')
plt.ylabel('Residuals')
plt.grid(True)
plt.show()

## Print model performance metrics
from sklearn.metrics import r2_score, mean_squared_error

r2 = r2_score(Y_test, y_pred)
mse = mean_squared_error(Y_test, y_pred)
rmse = np.sqrt(mse)

print(f"R-squared: {r2:.4f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"\nIntercept: {regressor.intercept_:.2f}")
print("Coefficients:")
for name, coef in zip(feature_names, coefficients):
    print(f"  {name}: {coef:.4f}")