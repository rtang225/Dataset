import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'initialexport.csv'
df = pd.read_csv(file_path)

# Filter for area > 10
df_filtered = df[df['area'] > 100000]
# df_filtered = df_filtered[df_filtered['area'] < 1000000]
print(df_filtered.size)

# Only show relationship with area (correlation of each feature with area)
feature_cols = [col for col in df_filtered.columns if col not in ['area', 'date', 'latitude', 'longitude', 'vNDVI', 'VARI']]
correlations = {}
for col in feature_cols:
    correlations[col] = df_filtered[col].corr(df_filtered['area'], method='pearson')

corr_df = pd.DataFrame.from_dict(correlations, orient='index', columns=['Correlation with Area'])

plt.figure(figsize=(8, 6))
sns.heatmap(corr_df, annot=True, fmt='.2f', cmap='coolwarm', cbar=True)
plt.title('Feature Correlation with Area (Pearson, Area > 10)')
plt.tight_layout()
plt.show()

# List features with correlation > 0.5 or < -0.5
strong_corr = corr_df[(corr_df['Correlation with Area'] > 0.5) | (corr_df['Correlation with Area'] < -0.5)]
if not strong_corr.empty:
    print("Features with strong correlation (>|0.5|) with area:")
    print(strong_corr)
else:
    print("No features with correlation > 0.5 or < -0.5 with area.")
