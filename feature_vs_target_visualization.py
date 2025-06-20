import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Load the dataset
file_path = 'initialexport.csv'
df = pd.read_csv(file_path)

# Remove only the top 5% of area values as outliers
upper_bound = df['area'].quantile(0.99)
df = df[df['area'] <= upper_bound]

# Assume 'area' is the target column
features = [col for col in df.columns if col != 'area']

# Create output directory if it doesn't exist
output_dir = 'Feature Graphs'
os.makedirs(output_dir, exist_ok=True)

# Plot each feature with normal distribution overlay
for col in features:
    if not np.issubdtype(df[col].dtype, np.number):
        continue  # Skip non-numeric columns
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=df[col], y=df['area'])
    plt.xlabel(col)
    plt.ylabel('area')
    mu = df[col].mean()
    sigma = df[col].std()
    print(f'{col} vs area\nMean: {mu:.4f}, Std: {sigma:.4f}')
    plt.title(f'{col} vs area\nMean: {mu:.4f}, Std: {sigma:.4f}')
    # Overlay normal distribution for the feature (mapped to x-axis)
    x_vals = np.linspace(df[col].min(), df[col].max(), 200)
    norm_pdf = (1/(sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_vals - mu)/sigma)**2)
    norm_pdf_scaled = norm_pdf * (df['area'].max() - df['area'].min()) / norm_pdf.max() + df['area'].min()
    plt.plot(x_vals, norm_pdf_scaled, 'r--', label=f'Normal Distribution ({col})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{col}_vs_area_normal99.png'))
    plt.close()