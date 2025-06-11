import pandas as pd
import numpy as np
import os
import cv2

data = {
    'Date': [],
    'Latitude': [],
    'Longtitude': [],
    'vNDVI': [],
    'VARI': [],
    'Area': []
}

# Read the CSV file and select only the required columns
df = pd.read_csv('data.csv', usecols=['DATE_DEBUT', 'SUP_HA', 'LATITUDE', 'LONGITUDE', 'CAUSE'])
folder = 'images\\wildfire'
counter = 0

# Access each value in the columns
for index, row in df.iterrows():
    date = row['DATE_DEBUT']
    area = row['SUP_HA']
    lat = row['LATITUDE']
    lon = row['LONGITUDE']
    cause = row['CAUSE']

    if cause == 'Humaine':
        filename = f"{lon},{lat}.jpg"
        filepath = os.path.join(folder, filename)
        if os.path.exists(filepath):
            data['Date'].append(date)
            data['Latitude'].append(lat)
            data['Longtitude'].append(lon)
            data['Area'].append(area)
            image_bgr = cv2.imread(filepath)
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
            counter += 1

# Create DataFrame and save to CSV
newdf = pd.DataFrame(data)
newdf.to_csv('humanwf.csv', index=False)