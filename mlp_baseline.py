import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

# Load the dataset
file_path = 'datasetreduced.csv'
df = pd.read_csv(file_path, usecols=['temperature_2m_mean','wind_speed_10m_max', 'relative_humidity_2m_mean', 'wind_speed_10m_mean', 'vapour_pressure_deficit_max', 'area_class', 'apparent_temperature_mean', 'rain_sum', 'soil_moisture_0_to_7cm_mean', 'soil_moisture_7_to_28cm_mean', 'dew_point_2m_mean'])
print(df.head())

# Prepare features and target
y = df['area_class']
X = df.drop(['area_class'], axis=1, errors='ignore')

# Encode target if not numeric
if y.dtype == 'O' or y.dtype.name == 'category':
    le = LabelEncoder()
    y = le.fit_transform(y)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Set device for CUDA usage
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Convert to torch tensors and move to device
torch_X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
torch_X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
torch_y_train = torch.tensor(y_train, dtype=torch.long).to(device)
torch_y_test = torch.tensor(y_test, dtype=torch.long).to(device)

# DataLoader
train_ds = TensorDataset(torch_X_train, torch_y_train)
test_ds = TensorDataset(torch_X_test, torch_y_test)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=32)

# Define a basic MLP model
class BasicMLP(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(num_features, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, num_classes)
    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

num_classes = len(np.unique(y))
num_features = X_train.shape[1]
model = BasicMLP(num_features, num_classes).to(device)

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0003)

# Training loop
num_epochs = 50
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    total = 0
    train_correct = 0
    train_total = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * xb.size(0)
        total += yb.size(0)
        preds = torch.argmax(out, dim=1)
        train_correct += (preds == yb).sum().item()
        train_total += yb.size(0)
    train_loss = running_loss / total
    train_acc = train_correct / train_total
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
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

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
    # Confusion matrix and classification report
    cm = confusion_matrix(y_test, preds)
    print('Confusion Matrix:')
    print(cm)
    print('Classification Report:')
    print(classification_report(y_test, preds, digits=3))

# Plot training and validation loss
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.axhline(y=acc, color='r', linestyle='--', label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training, Validation, and Test Accuracy')
plt.legend()
plt.tight_layout()
plt.show()
