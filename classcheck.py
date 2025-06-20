import pandas as pd

df = pd.read_csv('extremeclasses.csv')
print(df['area_class'].value_counts())