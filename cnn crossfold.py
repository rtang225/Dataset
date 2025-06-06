import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight
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

# Compute class weights for imbalanced classes
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

# Define CNN model
class SimpleCNN(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(2)
        self.flatten = nn.Flatten()
        # Dynamically determine the flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, 1, num_features)
            x = self.pool1(torch.relu(self.bn1(self.conv1(dummy))))
            x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
            x = self.pool3(torch.relu(self.bn3(self.conv3(x))))
            flat_size = x.numel()
        self.fc1 = nn.Linear(flat_size, 256)
        self.dropout1 = nn.Dropout(0.6)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, num_classes)
    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

num_classes = len(np.unique(y))
num_features = X_cnn.shape[2]
model = SimpleCNN(num_features, num_classes).to(device)
print(f"Model is on device: {next(model.parameters()).device}")

# Training setup
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# Training loop with 3-fold cross-validation
num_epochs = 20
kf = KFold(n_splits=3, shuffle=True, random_state=42)
fold_accuracies = []
all_train_losses = []
all_val_losses = []
all_train_accuracies = []
all_val_accuracies = []
for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"\nFold {fold+1}/3")
    # Split data for this fold
    X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
    y_train_fold, y_val_fold = y[train_idx], y[val_idx]
    # Fit scaler and encoder only on training fold
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_fold)
    X_val_scaled = scaler.transform(X_val_fold)
    # Reshape for Conv1D
    X_train_cnn = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
    X_val_cnn = X_val_scaled.reshape((X_val_scaled.shape[0], 1, X_val_scaled.shape[1]))
    # Convert to tensors
    torch_X_train = torch.tensor(X_train_cnn, dtype=torch.float32).to(device)
    torch_X_val = torch.tensor(X_val_cnn, dtype=torch.float32).to(device)
    torch_y_train = torch.tensor(y_train_fold, dtype=torch.long).to(device)
    torch_y_val = torch.tensor(y_val_fold, dtype=torch.long).to(device)
    train_ds = TensorDataset(torch_X_train, torch_y_train)
    val_ds = TensorDataset(torch_X_val, torch_y_val)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)
    model = SimpleCNN(num_features, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        total = 0
        for xb, yb in tqdm(train_loader, desc=f"Fold {fold+1} Epoch {epoch+1}/{num_epochs} [Train]"):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
            total += yb.size(0)
        train_loss = running_loss / total
        train_losses.append(train_loss)
        # Validation
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for xb, yb in tqdm(val_loader, desc=f"Fold {fold+1} Epoch {epoch+1}/{num_epochs} [Val]"):
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
        train_acc = None
        with torch.no_grad():
            train_correct = 0
            train_total = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                preds = torch.argmax(out, dim=1)
                train_correct += (preds == yb).sum().item()
                train_total += yb.size(0)
            train_acc = train_correct / train_total
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        if (epoch+1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    all_train_losses.append(train_losses)
    all_val_losses.append(val_losses)
    all_train_accuracies.append(train_accuracies)
    all_val_accuracies.append(val_accuracies)
    fold_accuracies.append(val_acc)
    print(f"Fold {fold+1} final validation accuracy: {val_acc:.4f}")
print(f"\nAverage 3-fold validation accuracy: {np.mean(fold_accuracies):.4f}")

# Plot training and validation loss/accuracy curves for each fold
plt.figure(figsize=(16, 6))
for i in range(len(all_train_losses)):
    plt.subplot(1, 2, 1)
    plt.plot(all_train_losses[i], label=f'Train Fold {i+1}')
    plt.plot(all_val_losses[i], label=f'Val Fold {i+1}', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(all_train_accuracies[i], label=f'Train Acc Fold {i+1}')
    plt.plot(all_val_accuracies[i], label=f'Val Acc Fold {i+1}', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
plt.tight_layout()
plt.show()

# Evaluation on test set
model.eval()
test_correct = 0
test_total = 0
with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb)
        preds = torch.argmax(out, dim=1)
        test_correct += (preds == yb).sum().item()
        test_total += yb.size(0)
test_acc = test_correct / test_total
print(f'Test accuracy: {test_acc:.3f}')

# Plot training, validation, and test accuracy on a line graph
plt.figure(figsize=(8, 6))
for i in range(len(all_train_accuracies)):
    plt.plot(all_train_accuracies[i], label=f'Train Acc Fold {i+1}')
    plt.plot(all_val_accuracies[i], label=f'Val Acc Fold {i+1}', linestyle='--')
plt.axhline(y=test_acc, color='r', linestyle='-', label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training, Validation, and Test Accuracy')
plt.legend()
plt.tight_layout()
plt.show()