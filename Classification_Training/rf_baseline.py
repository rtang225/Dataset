import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tqdm import tqdm
from imblearn.over_sampling import SMOTE

# Load the dataset
file_path = 'datasetreducedclasses.csv'
df = pd.read_csv(file_path)
print(df.head())

# Prepare features and target
y = df['area_class']
X = df.drop(['area_class', 'date', 'latitude', 'longitude', 'vNDVI', 'VARI'], axis=1, errors='ignore')

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

# Random Forest parameter tuning
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}
rf = RandomForestClassifier(class_weight='balanced', random_state=42)
grid = GridSearchCV(rf, param_grid, cv=5, verbose=2, n_jobs=-1)
print('Tuning Random Forest parameters...')
grid.fit(X_resampled, y_resampled)
print(f"Best Random Forest parameters: {grid.best_params_}")
print(f"Best Random Forest CV score: {grid.best_score_:.4f}")
preds = grid.predict(X_test)
acc = accuracy_score(y_test, preds)
print(f'Random Forest Test accuracy: {acc:.3f}')
print('Confusion Matrix:')
print(confusion_matrix(y_test, preds))
print('Classification Report:')
print(classification_report(y_test, preds, digits=3))
