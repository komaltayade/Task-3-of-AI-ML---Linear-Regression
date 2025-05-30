import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\Kamini Shewale\OneDrive\Desktop\AI-ML(internship)\task3\Housing.csv") 

print(df.columns)

X = df[['area']]      
y = df['price']    

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

print("Intercept:", model.intercept_)
print("Coefficient:", model.coef_)

y_pred = model.predict(X_test)

print("R² Score:", r2_score(y_test, y_pred))

plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', label='Predicted')
plt.title('Linear Regression: Actual vs Predicted')
plt.xlabel('Area')
plt.ylabel('Price')
plt.legend()
plt.show()
