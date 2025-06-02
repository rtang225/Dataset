import pandas as pd

# Read the CSV file
df = pd.read_csv('export.csv')

df = df.drop(columns = ['date'])
