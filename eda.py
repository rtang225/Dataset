import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import numpy as np

# Load the data
df = pd.read_csv('dataset.csv')

# Show basic info
df.info()
print(df.describe())
print(df.isnull().sum())

# Histograms for all numeric columns
df.hist(bins=30, figsize=(15, 10))
plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Check class distribution (class imbalance)
if 'area_class' in df.columns:
    print('Class distribution for area_class:')
    print(df['area_class'].value_counts())

# Perform PCA (excluding non-numeric columns and target)
features = df.select_dtypes(include=[np.number]).drop(columns=['area_class'], errors='ignore')
pca = PCA(n_components=5)
principal_components = pca.fit_transform(features)
print('Explained variance ratio for top 5 principal components:')
print(pca.explained_variance_ratio_)
# Optionally, add the principal components to the DataFrame
for i in range(principal_components.shape[1]):
    df[f'PC{i+1}'] = principal_components[:, i]