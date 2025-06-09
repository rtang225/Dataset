# requires some feature extraction from images, save for later testing?
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the data
csv_path = 'imagelabelsreduced.csv'
df = pd.read_csv(csv_path)

# Prepare features and target
y = df['Area_Class']
le = LabelEncoder()
y = le.fit_transform(y.astype(str))
X = df.drop(['Area_Class', 'Latitude', 'Longtitude', 'date'], axis=1, errors='ignore')

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Train Logistic Regression
clf = LogisticRegression(max_iter=1000, class_weight='balanced')
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f'Logistic Regression Test Accuracy: {acc:.4f}')
