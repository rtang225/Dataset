import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv('dataset.csv')

# Show basic info
df.info()
print(df.describe())
print(df.isnull().sum())

"""# Histograms for all numeric columns
df.hist(bins=30, figsize=(15, 10))
plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()"""

# Check class distribution (class imbalance)
if 'area_class' in df.columns:
    print('Class distribution for area_class:')
    print(df['area_class'].value_counts())