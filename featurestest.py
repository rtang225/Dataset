import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

data = {
    'Latitude': [],
    'Longtitude': [],
    'vNDVI': [],
    'VARI': [],
    'Class': []
}

folder_path = 'images\\nowildfire'
for filename in os.listdir(folder_path):
    path = os.path.join(folder_path, filename)
    if os.path.isfile(path):
        data['Latitude'].append(float(filename[:-4].split(',')[0]))
        data['Longtitude'].append(float(filename[:-4].split(',')[1]))
    image_bgr = cv2.imread(path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb = image_rgb.astype('float32') / 255.0  # Normalize to [0, 1]

    # Split channels
    R = image_rgb[:, :, 0]
    G = image_rgb[:, :, 1]
    B = image_rgb[:, :, 2]

    # Compute vNDVI and VARI with small epsilon to prevent division by zero
    epsilon = 1e-6
    vndvi = (G - R) / (G + R + epsilon)
    data['vNDVI'].append(np.mean(vndvi))
    vari = (G - R) / (G + R - B + epsilon)
    vari = np.clip(vari, -1, 1)  # Clip values to [-1, 1] for visualization
    data['VARI'].append(np.mean(vari))
    data['Class'].append(0)

folder_path = 'images\\wildfire'
for filename in os.listdir(folder_path):
    path = os.path.join(folder_path, filename)
    if os.path.isfile(path):
        data['Latitude'].append(float(filename[:-4].split(',')[0]))
        data['Longtitude'].append(float(filename[:-4].split(',')[1]))
    image_bgr = cv2.imread(path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb = image_rgb.astype('float32') / 255.0  # Normalize to [0, 1]

    # Split channels
    R = image_rgb[:, :, 0]
    G = image_rgb[:, :, 1]
    B = image_rgb[:, :, 2]

    # Compute vNDVI and VARI with small epsilon to prevent division by zero
    epsilon = 1e-6
    vndvi = (G - R) / (G + R + epsilon)
    data['vNDVI'].append(np.mean(vndvi))
    vari = (G - R) / (G + R - B + epsilon)
    vari = np.clip(vari, -1, 1)  # Clip values to [-1, 1] for visualization
    data['VARI'].append(np.mean(vari))
    data['Class'].append(1)


# Create DataFrame and save to CSV
df = pd.DataFrame(data)
df.to_csv('features.csv', index=False)

"""
# Load RGB image and convert to float
image_bgr = cv2.imread('test10.png')  # Replace with your image path
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
image_rgb = image_rgb.astype('float32') / 255.0  # Normalize to [0, 1]

# Split channels
R = image_rgb[:, :, 0]
G = image_rgb[:, :, 1]
B = image_rgb[:, :, 2]

# Compute vNDVI and VARI with small epsilon to prevent division by zero
epsilon = 1e-6
vndvi = (G - R) / (G + R + epsilon)
vari = (G - R) / (G + R - B + epsilon)
vari = np.clip(vari, -1, 1)  # Clip values to [-1, 1] for visualization

# Plot results with colorbars to show values
plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
plt.imshow(image_rgb)
plt.title("Original RGB")
plt.axis('off')

plt.subplot(1, 3, 2)
vndvi_img = plt.imshow(vndvi, cmap='RdYlGn', vmin=-1, vmax=1)
plt.title(f"vNDVI\navg={np.mean(vndvi):.4f}")
plt.colorbar(vndvi_img, fraction=0.046, pad=0.04)
plt.axis('off')

plt.subplot(1, 3, 3)
vari_img = plt.imshow(vari, cmap='RdYlGn', vmin=-1, vmax=1)
plt.title(f"VARI\navg={np.mean(vari):.4f}")
plt.colorbar(vari_img, fraction=0.046, pad=0.04)
plt.axis('off')

plt.tight_layout()
plt.show()
"""