import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelBinarizer

# Load the data
file_path = 'imagelabelsreduced.csv'
df = pd.read_csv(file_path)

# Dummy (one-hot) encode the target
y = df['Area_Class'].fillna('Unknown').astype(str)
lb = LabelBinarizer()
y_dummy = lb.fit_transform(y)
print('Dummy-encoded target shape:', y_dummy.shape)
print('Classes:', lb.classes_)

# Train/test split
_, y_test = train_test_split(y, test_size=0.2, random_state=42, stratify=y)

# Find the most common class in the training set
y_train, _ = train_test_split(y, test_size=0.2, random_state=42, stratify=y)
most_common_class = y_train.value_counts().idxmax()

# Predict the most common class for all test samples
baseline_preds = [most_common_class] * len(y_test)

# Calculate baseline accuracy
baseline_acc = accuracy_score(y_test, baseline_preds)
print(f"Baseline (most common class) accuracy: {baseline_acc:.4f}")
