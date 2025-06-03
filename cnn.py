import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'dataset.csv'
df = pd.read_csv(file_path)

# Prepare features and target
y = df['area_class']
X = df.drop(['area_class', 'date', 'latitude', 'longitude', 'rain_sum', 'precipitation_sum', 'temperature_2m_mean', 'growing_degree_days_base_0_limit_50', 'apparent_temperature_mean', 'wet_bulb_temperature_2m_mean', 'et0_fao_evapotranspiration_sum', 'dew_point_2m_mean', 'wind_gusts_10m_max', 'et0_fao_evapotranspiration', 'soil_temperature_0_to_7cm_mean'], axis=1, errors='ignore')

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

# Define CNN model
class SimpleCNN(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(num_features * 64, 128)
        # self.dropout1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, 64)
        # self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, num_classes)
    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        # x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        # x = self.dropout2(x)
        x = self.fc3(x)
        return x

num_classes = len(np.unique(y))
num_features = X_cnn.shape[2]
model = SimpleCNN(num_features, num_classes).to(device)
print(f"Model is on device: {next(model.parameters()).device}")

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)

# Training loop
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * xb.size(0)
        preds = torch.argmax(out, dim=1)
        correct += (preds == yb).sum().item()
        total += yb.size(0)
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)

    # Validation
    model.eval()
    val_running_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
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
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

# Plot learning and accuracy curves
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(range(1, num_epochs+1), train_losses, marker='o', label='Train Loss')
plt.plot(range(1, num_epochs+1), val_losses, marker='x', label='Val Loss')
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.subplot(1,2,2)
plt.plot(range(1, num_epochs+1), train_accuracies, marker='o', label='Train Acc')
plt.plot(range(1, num_epochs+1), val_accuracies, marker='x', label='Val Acc')
plt.title('Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.show()

# Evaluation
model.eval()
with torch.no_grad():
    preds = []
    for xb, _ in test_loader:
        xb = xb.to(device)
        out = model(xb)
        preds.append(torch.argmax(out, dim=1).cpu().numpy())
    preds = np.concatenate(preds)
    acc = accuracy_score(y_test, preds)
    print(f'Test accuracy: {acc:.3f}')
