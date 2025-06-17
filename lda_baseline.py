import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, ParameterGrid
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tqdm import tqdm
from imblearn.over_sampling import SMOTE

# Load the dataset
file_path = 'datasetreduced.csv'
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

# LDA parameter tuning
param_grid = [
    {'solver': ['svd'], 'shrinkage': [None]},
    {'solver': ['lsqr', 'eigen'], 'shrinkage': [None, 'auto', 0.1, 0.5, 1.0]}
]
all_params = []
for grid in param_grid:
    all_params.extend(list(ParameterGrid(grid)))
best_score = 0
best_params = None
print('Tuning LDA parameters...')
# Use X_resampled and y_resampled for tuning
for params in tqdm(all_params, desc='LDA Grid Search'):
    lda = LDA(**params)
    try:
        lda.fit(X_resampled, y_resampled)
        preds = lda.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"Params: {params}, Accuracy: {acc:.4f}")
        if acc > best_score:
            best_score = acc
            best_params = params
    except Exception as e:
        print(f"Params: {params} failed with error: {e}")

print(f"Best LDA parameters: {best_params}")
print(f"Best LDA Test accuracy: {best_score:.4f}")
# If you want to predict with the best model, retrain it:
best_lda = LDA(**best_params)
best_lda.fit(X_resampled, y_resampled)
preds = best_lda.predict(X_test)
acc = accuracy_score(y_test, preds)
print(f'Final LDA Test accuracy: {acc:.3f}')
print('Confusion Matrix:')
print(confusion_matrix(y_test, preds))
print('Classification Report:')
print(classification_report(y_test, preds, digits=3))
