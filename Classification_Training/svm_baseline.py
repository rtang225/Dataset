import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tqdm import tqdm
from imblearn.over_sampling import SMOTE

# Load the dataset
file_path = 'datasetreduced.csv'
df = pd.read_csv(file_path, usecols=['temperature_2m_mean','wind_speed_10m_max', 'relative_humidity_2m_mean', 'wind_speed_10m_mean', 'vapour_pressure_deficit_max', 'area_class', 'apparent_temperature_mean', 'rain_sum', 'soil_moisture_0_to_7cm_mean', 'soil_moisture_7_to_28cm_mean', 'dew_point_2m_mean', 'vNDVI', 'VARI', 'wind_gusts_10m_max', 'wind_gusts_10m_mean', 'soil_moisture_0_to_100cm_mean', 'wet_bulb_temperature_2m_mean'])#])
print(df.head())

# Prepare features and target
y = df['area_class']
X = df.drop(['area_class'], axis=1, errors='ignore')
"""
# Prepare features and target
y = df['area_class']
X = df.drop(['area_class', 'date', 'latitude', 'longitude', 'vNDVI', 'VARI'], axis=1, errors='ignore')
"""
# Encode target if not numeric
if y.dtype == 'O' or y.dtype.name == 'category':
    le = LabelEncoder()
    y = le.fit_transform(y)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Apply SMOTE to balance classes
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Train SVM
print('Training SVM...')
clf = SVC(C=100, kernel='rbf', gamma='auto', class_weight='balanced', probability=True)
clf.fit(X_train, y_train)
preds = clf.predict(X_test)
acc = accuracy_score(y_test, preds)
print(f"Accuracy: {acc:.4f}")

# Evaluate
best_clf = SVC(C=100, kernel='rbf', gamma='auto', class_weight='balanced', probability=True)
best_clf.fit(X_train, y_train)
preds = best_clf.predict(X_test)
acc = accuracy_score(y_test, preds)
print(f'Test accuracy: {acc:.3f}')
print('Confusion Matrix:')
print(confusion_matrix(y_test, preds))
print('Classification Report:')
print(classification_report(y_test, preds, digits=3))
