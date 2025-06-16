import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

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

# Train logistic regression
clf = LogisticRegression(max_iter=1000, class_weight='balanced')
clf.fit(X_train, y_train)

# Evaluate
preds = clf.predict(X_test)
acc = accuracy_score(y_test, preds)
print(f'Test accuracy: {acc:.3f}')
print('Confusion Matrix:')
print(confusion_matrix(y_test, preds))
print('Classification Report:')
print(classification_report(y_test, preds, digits=3))
