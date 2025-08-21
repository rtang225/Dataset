import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tqdm import tqdm
from imblearn.over_sampling import SMOTE

# Load the dataset
file_path = 'datasetreducedclasses.csv'
df = pd.read_csv(file_path, usecols=['temperature_2m_mean','wind_speed_10m_max', 'relative_humidity_2m_mean', 'wind_speed_10m_mean', 'vapour_pressure_deficit_max', 'area_class', 'apparent_temperature_mean', 'rain_sum', 'soil_moisture_0_to_7cm_mean', 'soil_moisture_7_to_28cm_mean', 'dew_point_2m_mean'])#, 'wind_gusts_10m_max', 'wind_gusts_10m_mean', 'soil_moisture_0_to_100cm_mean', 'wet_bulb_temperature_2m_mean'])#, 'vNDVI', 'VARI'])
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

# XGBoost parameter tuning
param_grid_xgb = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}
xgb = XGBClassifier(eval_metric='mlogloss', random_state=42)
grid_xgb = GridSearchCV(xgb, param_grid_xgb, cv=5, verbose=2, n_jobs=-1)
print('Tuning XGBoost parameters...')
grid_xgb.fit(X_resampled, y_resampled)
print(f"Best XGBoost parameters: {grid_xgb.best_params_}")
print(f"Best XGBoost CV score: {grid_xgb.best_score_:.4f}")
preds = grid_xgb.predict(X_test)
acc = accuracy_score(y_test, preds)
print(f'XGBoost Test accuracy: {acc:.3f}')
print('Confusion Matrix:')
print(confusion_matrix(y_test, preds))
print('Classification Report:')
print(classification_report(y_test, preds, digits=3))

print('Training XGBoost...')
xgb = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, subsample=1.0, colsample_bytree=1.0, eval_metric='mlogloss', random_state=42, tree_method='hist')
print('XGBoost will use device: cpu')
xgb.fit(X_resampled, y_resampled)
