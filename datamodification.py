import pandas as pd

# Read the CSV file
# df = pd.read_csv('initialexport.csv')
df = pd.read_csv('initialexport.csv', usecols=['latitude', 'longitude', 'area'])

# Reclassify 'area' variable into area_class
"""
bins = [0, 0.1, 1.0, 10, 100, 1000, 10000, 100000, float('inf')]
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
"""

"""
bins = [0, 100, 1000, float('inf')]
labels = ['A', 'B', 'C']
df['area_class'] = pd.cut(df['area'], bins=bins, labels=labels, right=True)
df = df.drop(columns=['area'])
"""

df.to_csv('regimagelabels.csv', index=False)