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
    # Overlay linear, quadratic, cubic, and logarithmic fits
    try:
        # Linear fit
        coeffs_lin = np.polyfit(df[col], df['area'], 1)
        lin_fit = np.poly1d(coeffs_lin)
        plt.plot(df[col], lin_fit(df[col]), 'b--', label='Linear Fit')
        # Quadratic fit
        coeffs_quad = np.polyfit(df[col], df['area'], 2)
        quad_fit = np.poly1d(coeffs_quad)
        plt.plot(df[col], quad_fit(df[col]), 'm--', label='Quadratic Fit')
        # Cubic fit
        coeffs_cubic = np.polyfit(df[col], df['area'], 3)
        cubic_fit = np.poly1d(coeffs_cubic)
        plt.plot(df[col], cubic_fit(df[col]), 'c--', label='Cubic Fit')
        # Logarithmic fit (only if all x > 0)
        if (df[col] > 0).all():
            log_x = np.log(df[col])
            coeffs_log = np.polyfit(log_x, df['area'], 1)
            log_fit = np.poly1d(coeffs_log)
            plt.plot(df[col], log_fit(np.log(df[col])), 'g--', label='Logarithmic Fit')
        # Exponential fit (only if all y > 0)
        if (df['area'] > 0).all():
            log_y = np.log(df['area'])
            coeffs_exp = np.polyfit(df[col], log_y, 1)
            exp_fit = lambda x: np.exp(coeffs_exp[1]) * np.exp(coeffs_exp[0] * x)
            plt.plot(df[col], exp_fit(df[col]), 'y--', label='Exponential Fit')
    except Exception as e:
        print(f'Could not fit curve for {col}: {e}')
    plt.legend()
    plt.tight_layout()
    plt.show()
    # plt.savefig(os.path.join(output_dir, f'{col}_vs_area_normal99.png'))
    # plt.close()