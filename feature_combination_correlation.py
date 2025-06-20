import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from itertools import combinations

# Load the dataset
file_path = 'initialexport.csv'
df = pd.read_csv(file_path)

# Combine features (example: sum, product, ratio, etc.)
combined_features = pd.DataFrame()
features = [col for col in df.columns if col != 'area']

# Only use numeric columns for combinations
numeric_features = [col for col in features if np.issubdtype(df[col].dtype, np.number)]

# Example combinations (customize as needed)
comb_dict = {}
# Pairwise combinations (already present)
for i, col1 in enumerate(tqdm(numeric_features, desc='Combining features (pairs and triples)')):
    for j, col2 in enumerate(numeric_features):
        if i < j:
            comb_dict[f'{col1}_plus_{col2}'] = df[col1] + df[col2]
            comb_dict[f'{col1}_times_{col2}'] = df[col1] * df[col2]
            # Avoid division by zero
            safe_col2 = df[col2].replace(0, np.nan)
            comb_dict[f'{col1}_div_{col2}'] = df[col1] / safe_col2
            comb_dict[f'{col1}_minus_{col2}'] = df[col1] - df[col2]
            comb_dict[f'{col1}_mean_{col2}'] = (df[col1] + df[col2]) / 2
            comb_dict[f'{col1}_max_{col2}'] = np.maximum(df[col1], df[col2])
            comb_dict[f'{col1}_min_{col2}'] = np.minimum(df[col1], df[col2])
# Triple combinations (sum, product, mean)
for cols in combinations(numeric_features, 3):
    c1, c2, c3 = cols
    comb_dict[f'{c1}_plus_{c2}_plus_{c3}'] = df[c1] + df[c2] + df[c3]
    comb_dict[f'{c1}_times_{c2}_times_{c3}'] = df[c1] * df[c2] * df[c3]
    comb_dict[f'{c1}_mean_{c2}_{c3}'] = (df[c1] + df[c2] + df[c3]) / 3
combined_features = pd.concat(comb_dict, axis=1)
combined_features['area'] = df['area']

# Compute correlation matrix
corr = combined_features.corr()

"""# Plot heatmap of correlations with area
plt.figure(figsize=(12, 8))
area_corr = corr['area'].drop('area').sort_values(ascending=False)
sns.barplot(x=area_corr.values, y=area_corr.index)
plt.title('Correlation of Combined Features with Area')
plt.xlabel('Correlation Coefficient')
plt.tight_layout()
plt.show()
"""
# Optionally, save the correlations to a CSV
combined_features.corr()['area'].sort_values(ascending=False).to_csv('combined_feature_area_correlations.csv')
