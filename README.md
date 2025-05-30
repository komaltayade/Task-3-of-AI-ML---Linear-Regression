# Task 3: Linear Regression – AI/ML Internship

## 📌 Objective:
To build a simple linear regression model to predict target values and understand model performance.

---

## 📂 Dataset:
- **Dataset used:** House Price Prediction Dataset
- **Example Features:** Area, Bedrooms, Bathrooms
- **Target:** House Price

---

## 🛠️ Tools Used:
- Python
- Pandas
- Scikit-learn (sklearn)
- Matplotlib

---

## 🧪 Steps Performed:

### 1. Import and Preprocess the Dataset
- Loaded the dataset using Pandas.
- Checked for missing values.
- Cleaned the dataset by dropping or filling missing values.

### 2. Split Data into Train-Test Sets
- Divided the data into training (80%) and testing (20%) sets using train_test_split.

### 3. Fit a Linear Regression Model
- Used LinearRegression() from sklearn.
- Trained the model on training data (X_train and y_train).

### 4. Evaluate the Model
- Predicted values on the test set.
- Calculated evaluation metrics:
  - **MAE (Mean Absolute Error)**
  - **MSE (Mean Squared Error)**
  - **R² (R-Squared Score)**

### 5. Plot Regression Line
- Plotted actual vs predicted values.
- Visualized how well the regression line fits the data.

---

## 📈 Results:
- Regression line plotted correctly.
- Model gave a reasonable R² score indicating the model fit.
- Errors (MAE and MSE) calculated to understand how close predictions were.

---

## ❓ Interview Questions:

### 1. What assumptions does linear regression make?
- The relationship between input and output is **linear**.
- Errors (residuals) should be **normally distributed**.
- No strong **multicollinearity** between independent variables.
- **Homoscedasticity**: Error variance should be constant.

### 2. How do you interpret the coefficients?
- The **coefficient** tells how much the target value (like Price) will change for every unit change in the feature (like Area).
- Example: If coefficient = 2000, then for every 1 sq ft increase in Area, Price increases by 2000.

### 3. What is R² score and its significance?
- **R² score** tells how much of the variance in the target is explained by the model.
- **Closer to 1** = better model. **Closer to 0** = poor model.

### 4. When would you prefer MSE over MAE?
- Use **MSE** when you want to **punish large errors** more strongly.
- Use **MAE** when you want to **treat all errors equally**.

### 5. How do you detect multicollinearity?
- By checking **correlation matrix** or using **VIF (Variance Inflation Factor)**.
- High correlation between features means multicollinearity.

### 6. What is the difference between simple and multiple regression?
- **Simple regression** = 1 independent feature.
- **Multiple regression** = 2 or more independent features.

### 7. Can linear regression be used for classification?
- **No**, it is meant for predicting continuous values.
- For classification, use models like Logistic Regression.

### 8. What happens if you violate regression assumptions?
- The model will give **biased** or **incorrect** results.
- Predictions may not be reliable.

---

## ✅ Completed By:
**Kamini Shewale**  
AI & ML Internship – Elevate Labs  
Date: *[29-04-2025]*