import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Your existing data preparation code
dataset = pd.read_csv("HousingData.csv")
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values
corelation_matrix = dataset.corr()

imputer = SimpleImputer(strategy="mean", missing_values=np.nan)
X = imputer.fit_transform(X)

x_train, x_test, y_train, y_test = train_test_split(X[:,5], Y, train_size=0.8, random_state=0)

# Model training
regressor = LinearRegression()
regressor.fit(X=x_train.reshape(-1, 1), y=y_train)

# Make predictions
y_predict = regressor.predict(x_test.reshape(-1, 1))

# Calculate performance metrics
r2 = r2_score(y_test, y_predict)
mse = mean_squared_error(y_test, y_predict)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_predict)

print(f"R-squared Score: {r2:.4f}")
print(f"Mean Squared Error: {mse:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")

#Heat Graph
plt.figure(figsize=(12,10))
sb.heatmap(corelation_matrix,annot=True,cmap="coolwarm",vmin=-1,vmax=1,center=0)
plt.show()

# Visualization with performance metrics
plt.figure(figsize=(10, 6))
plt.scatter(x_train, y_train, color="red", label="Training Data")
plt.plot(x_train, regressor.predict(x_train.reshape(-1, 1)), color="blue", label="Regression Line")
plt.title(f"Linear Regression (R² = {r2:.4f})")
plt.xlabel("Feature")
plt.ylabel("Target")
plt.legend()
plt.show()

# Add residual plot
plt.figure(figsize=(10, 6))
residuals = y_test - y_predict
plt.scatter(y_predict, residuals, color="green")
plt.axhline(y=0, color="red", linestyle="--")
plt.title("Residual Plot")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.show()