
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load dataset
df = pd.read_csv("Housing.csv")

# Convert any categorical columns if needed
# Example: if 'mainroad', 'furnishingstatus', etc., are in dataset
df = pd.get_dummies(df, drop_first=True)

# 2. Simple Linear Regression using one feature (e.g., 'area')
X_simple = df[['area']]
y = df['price']

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_simple, y, test_size=0.2, random_state=42)

# Train model
model_simple = LinearRegression()
model_simple.fit(X_train_s, y_train_s)
y_pred_s = model_simple.predict(X_test_s)

# Evaluation - Simple
print("Simple Linear Regression (area -> price):")
print("Coefficient:", model_simple.coef_[0])
print("Intercept:", model_simple.intercept_)
print("MAE:", mean_absolute_error(y_test_s, y_pred_s))
print("MSE:", mean_squared_error(y_test_s, y_pred_s))
print("R² Score:", r2_score(y_test_s, y_pred_s))

# Plot regression line
plt.figure(figsize=(8, 5))
plt.scatter(X_test_s, y_test_s, color='blue', label='Actual Price')
plt.plot(X_test_s, y_pred_s, color='red', linewidth=2, label='Predicted Price')
plt.xlabel('Area')
plt.ylabel('Price')
plt.title('Simple Linear Regression')
plt.legend()
plt.grid(True)
plt.show()

# 3. Multiple Linear Regression using all features
X_multi = df.drop('price', axis=1)
y_multi = df['price']

X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
    X_multi, y_multi, test_size=0.2, random_state=42)

# Train model
model_multi = LinearRegression()
model_multi.fit(X_train_m, y_train_m)
y_pred_m = model_multi.predict(X_test_m)

# Evaluation - Multiple
print("\nMultiple Linear Regression (all features -> price):")
print("Intercept:", model_multi.intercept_)
print("MAE:", mean_absolute_error(y_test_m, y_pred_m))
print("MSE:", mean_squared_error(y_test_m, y_pred_m))
print("R² Score:", r2_score(y_test_m, y_pred_m))

# Show coefficients
coef_df = pd.DataFrame({
    'Feature': X_multi.columns,
    'Coefficient': model_multi.coef_
})
print("\nFeature Coefficients:\n", coef_df)
