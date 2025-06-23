import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('initialexport.csv')

# Print each y value and its log1p value
for val in df['area']:
    print(f'Original: {val}, log1p: {np.log1p(val)}')
