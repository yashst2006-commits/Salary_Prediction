import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 1. Load & clean dataset
df = pd.read_csv("Salary_Prediction/dataset/Salary Data.csv")

# Drop missing values
df = df.dropna()

# One-hot encoding for categorical features
df = pd.get_dummies(
    df,
    columns=["Gender", "Education Level", "Job Title"],
    drop_first=True
)

# 2. LINEAR REGRESSION
X = df.drop("Salary", axis=1)
y = df["Salary"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)

print("Linear Regression")
print("MAE:", mean_absolute_error(y_test, y_pred_lr))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_lr)))
print("R2:", r2_score(y_test, y_pred_lr))

# 3. POLYNOMIAL REGRESSION
# (Numeric feature only)
X_poly = df[["Years of Experience"]]
y_poly = df["Salary"]

X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
    X_poly, y_poly, test_size=0.2, random_state=42
)

poly = PolynomialFeatures(degree=3)
X_train_poly = poly.fit_transform(X_train_p)
X_test_poly = poly.transform(X_test_p)

poly_lr = LinearRegression()
poly_lr.fit(X_train_poly, y_train_p)

y_pred_poly = poly_lr.predict(X_test_poly)

print("\nPolynomial Regression (Degree 3)")
print("MAE:", mean_absolute_error(y_test_p, y_pred_poly))
print("RMSE:", np.sqrt(mean_squared_error(y_test_p, y_pred_poly)))
print("R2:", r2_score(y_test_p, y_pred_poly))

# 4. VISUALIZATION
X_vis = df[["Years of Experience"]]
y_vis = df["Salary"]

X_range = np.linspace(
    X_vis.min().values[0],
    X_vis.max().values[0],
    100
).reshape(-1, 1)

X_range_df = pd.DataFrame(X_range, columns=["Years of Experience"])
X_range_poly = poly.transform(X_range_df)

y_poly_curve = poly_lr.predict(X_range_poly)

plt.figure(figsize=(8, 5))
plt.scatter(X_vis, y_vis, color="blue", label="Actual Data")
plt.plot(X_range, y_poly_curve, color="red", linewidth=2, label="Polynomial Regression")

plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Salary Prediction using Polynomial Regression")
plt.legend()
plt.savefig("Salary_Prediction/images/Salary_pred_Poly_Reg.png")
plt.show()
