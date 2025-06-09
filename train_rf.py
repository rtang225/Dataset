import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the data
csv_path = 'imagelabelsreduced.csv'
df = pd.read_csv(csv_path)

# Prepare features and target
y = df['Area_Class']
le = LabelEncoder()
y = le.fit_transform(y.astype(str))
X = df.drop(['Area_Class', 'Latitude', 'Longtitude', 'date'], axis=1, errors='ignore')

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train Random Forest
clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f'Random Forest Test Accuracy: {acc:.4f}')
