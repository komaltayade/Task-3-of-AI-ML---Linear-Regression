#Split data into train-test sets.
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv(r"C:\Users\Kamini Shewale\OneDrive\Desktop\AI-ML(internship)\task3\Housing.csv") 
print(df.columns)
# Example: predicting 'Price' based on 'Area'
X = df[['area']]      
y = df['price']    

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
