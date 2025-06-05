import pandas as pd

# Read the CSV file
df = pd.read_csv('wf.csv')
df = df.drop(columns = ['Date', 'vNDVI', 'VARI'])

# Reclassify 'area' variable into area_class
bins = [0, 0.1, 1.0, 10, 100, 1000, 10000, 100000, float('inf')]
labels = ['Type A', 'Type B', 'Type C', 'Type D', 'Type E', 'Type F', 'Type G', 'Type H']
df['Area_Class'] = pd.cut(df['Area'], bins=bins, labels=labels, right=True)
df = df.drop(columns=['Area'])

df.to_csv('image_labels.csv', index=False)