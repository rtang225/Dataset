import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Load the dataset
file_path = 'initialexport.csv'
df = pd.read_csv(file_path, usecols=['temperature_2m_mean','wind_speed_10m_max', 'relative_humidity_2m_mean', 'wind_speed_10m_mean', 'vapour_pressure_deficit_max', 'area', 'apparent_temperature_mean', 'vNDVI', 'VARI'])#, 'rain_sum', 'soil_moisture_0_to_7cm_mean', 'soil_moisture_7_to_28cm_mean', 'dew_point_2m_mean'])#, 'wind_gusts_10m_max', 'wind_gusts_10m_mean', 'soil_moisture_0_to_100cm_mean', 'wet_bulb_temperature_2m_mean'])#])
print(df.head())

# Encode non-numeric columns (except target)
for col in df.columns:
    if df[col].dtype == 'O' and col != 'area':
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# Assume 'area' is the target column (change if needed)
X = df.drop(['area'], axis=1, errors='ignore')
y = df['area']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Apply bins and labels to y_test only
bins = [0, 0.1, 1.0, 10, 100, 1000, 10000, 100000, float('inf')]
labels = [0, 1, 2, 3, 4, 5, 6, 7]
y_test = pd.cut(y_test, bins=bins, labels=labels, include_lowest=True)

# Convert to torch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

train_ds = TensorDataset(X_train_tensor, y_train_tensor)
test_ds = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=32)

# Define a simple neural network regressor
class SimpleRegressor(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.fc1 = nn.Linear(num_features, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 1)
    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

num_features = X_train.shape[1]
model = SimpleRegressor(num_features)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)

num_epochs = 20
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * xb.size(0)
    train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(train_loss)
    # Validation
    model.eval()
    val_running_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            out = model(xb)
            loss = criterion(out, yb)
            val_running_loss += loss.item() * xb.size(0)
            preds = torch.argmax(out, dim=1)
            val_correct += (preds == yb).sum().item()
            val_total += yb.size(0)
    val_loss = val_running_loss / val_total
    val_acc = val_correct / val_total
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)
    if (epoch+1) % 5 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

# Final evaluation
model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor).cpu().numpy().flatten()
    y_pred = pd.cut(y_pred, bins=bins, labels=labels, include_lowest=True)
    print(y_test)
    print(y_pred)
    acc = accuracy_score(y_test, y_pred)
    print(f'Test accuracy: {acc:.3f}')
    # Confusion matrix and classification report
    cm = confusion_matrix(y_test, y_pred)
    print('Confusion Matrix:')
    print(cm)
    print('Classification Report:')
    print(classification_report(y_test, y_pred, digits=3))
    print(np.bincount(y_pred))

# Plot training and validation loss
plt.figure(figsize=(8,5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Neural Network Regression Training and Validation Loss')
plt.legend()
plt.tight_layout()
plt.show()
