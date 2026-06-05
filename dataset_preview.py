import pandas as pd 
df = pd.read_csv('train.csv')
print(df.head())
print()
print(df.columns)
print()
print(df.describe())
print()
print(df.info())

