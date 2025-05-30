#5.Plot regression line and interpret coefficients.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\Kamini Shewale\OneDrive\Desktop\AI-ML(internship)\task3\Housing.csv") 

print(df.columns)

X = df[['area']]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"R² Score: {r2}")

print(f"Intercept (b): {model.intercept_}")
print(f"Coefficient (slope) for area: {model.coef_[0]}")

plt.scatter(X_test['area'], y_test, color='blue', label='Actual')
plt.plot(X_test['area'], y_pred, color='red', linewidth=2, label='Predicted Line')
plt.xlabel("Area")
plt.ylabel("Price")
plt.title("Area vs Price (Linear Regression)")
plt.legend()
plt.show()

