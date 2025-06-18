import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'datasetreduced.csv'
df = pd.read_csv(file_path)

# Convert area_class to numeric if it contains A, B, C, D
if df['area_class'].dtype == 'O' or df['area_class'].dtype.name == 'category':
    df['area_class'] = df['area_class'].map({'A': 1, 'B': 2, 'C': 3, 'D': 4})

# Drop non-feature columns for correlation
feature_cols = [col for col in df.columns if col not in ['area_class', 'date', 'latitude', 'longitude', 'vNDVI', 'VARI']]
features = df[feature_cols]

# Safely add area_class to features DataFrame
features = features.copy()
features['area_class'] = df['area_class']

# Compute Pearson correlation matrix
corr_matrix = features.corr(method='pearson')

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True)
plt.title('Feature and Target Correlation Heatmap (Pearson)')
plt.tight_layout()
plt.show()

# Optionally, compute and plot polynomial (Spearman) correlation
spearman_corr = features.corr(method='spearman')
plt.figure(figsize=(12, 10))
sns.heatmap(spearman_corr, annot=True, fmt='.2f', cmap='viridis', square=True)
plt.title('Feature and Target Correlation Heatmap (Spearman)')
plt.tight_layout()
plt.show()
