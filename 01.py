import pandas as pd

df = pd.read_csv(r"C:\Users\Kamini Shewale\OneDrive\Desktop\AI-ML(internship)\task3\Housing.csv") 

print(df.isnull().sum())

df = df.dropna()
