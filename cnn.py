import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Load the dataset
file_path = 'dataset.csv'
df = pd.read_csv(file_path)

# Prepare features and target
y = df['area_class']
X = df.drop(['area_class', 'date', 'latitude', 'longitude'], axis=1, errors='ignore')

# Encode target if not numeric
if y.dtype == 'O' or y.dtype.name == 'category':
    le = LabelEncoder()
    y = le.fit_transform(y)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reshape for Conv1D: (samples, channels, timesteps)
X_cnn = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_cnn, y, test_size=0.2, random_state=42)

# Convert to torch tensors
torch_X_train = torch.tensor(X_train, dtype=torch.float32)
torch_X_test = torch.tensor(X_test, dtype=torch.float32)
torch_y_train = torch.tensor(y_train, dtype=torch.long)
torch_y_test = torch.tensor(y_test, dtype=torch.long)

# DataLoader
train_ds = TensorDataset(torch_X_train, torch_y_train)
test_ds = TensorDataset(torch_X_test, torch_y_test)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=32)

# Define CNN model
class SimpleCNN(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear((num_features-2)*32, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, num_classes)
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

num_classes = len(np.unique(y))
num_features = X_cnn.shape[2]
model = SimpleCNN(num_features, num_classes)

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(20):
    model.train()
    for xb, yb in train_loader:
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
    if (epoch+1) % 5 == 0:
        print(f"Epoch {epoch+1}/20, Loss: {loss.item():.4f}")

# Evaluation
model.eval()
with torch.no_grad():
    preds = []
    for xb, _ in test_loader:
        out = model(xb)
        preds.append(torch.argmax(out, dim=1).cpu().numpy())
    preds = np.concatenate(preds)
    acc = accuracy_score(y_test, preds)
    print(f'Test accuracy: {acc:.3f}')
