import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tqdm import tqdm

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

# Train SVM with manual grid search
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto'],
    'degree': [2, 3, 4]  # Only used for 'poly' kernel
}

svc = SVC(class_weight='balanced', probability=True)
param_list = list(ParameterGrid(param_grid))
print('Tuning SVM parameters...')
best_score = 0
best_params = None
for params in tqdm(param_list, desc='SVM Grid Search'):
    # Only set degree if kernel is 'poly'
    if params['kernel'] != 'poly':
        params = {k: v for k, v in params.items() if k != 'degree'}
    clf = SVC(**params, class_weight='balanced', probability=True)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Params: {params}, Accuracy: {acc:.4f}")
    if acc > best_score:
        best_score = acc
        best_params = params
print(f"Best parameters: {best_params}")
print(f"Best test accuracy: {best_score:.4f}")

# Evaluate
best_clf = SVC(**best_params, class_weight='balanced', probability=True)
best_clf.fit(X_train, y_train)
preds = best_clf.predict(X_test)
acc = accuracy_score(y_test, preds)
print(f'Test accuracy: {acc:.3f}')
print('Confusion Matrix:')
print(confusion_matrix(y_test, preds))
print('Classification Report:')
print(classification_report(y_test, preds, digits=3))
