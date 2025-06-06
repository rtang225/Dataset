import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

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

# Compute class weights for imbalanced classes
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
class_weights = torch.tensor(class_weights, dtype=torch.float32)

# Hyperparameter grid
learning_rates = [0.001, 0.0005, 0.0001]
batch_sizes = [16, 32, 64, 128]
num_filters_list = [16, 32, 64, 128, 256]

# Train/val split
X_train, X_val, y_train, y_val = train_test_split(X_cnn, y, test_size=0.2, random_state=42, stratify=y)

# CNN model definition
def make_cnn(num_features, num_classes, num_filters):
    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv1d(1, num_filters, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm1d(num_filters)
            self.pool1 = nn.MaxPool1d(2)
            self.conv2 = nn.Conv1d(num_filters, num_filters*2, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm1d(num_filters*2)
            self.pool2 = nn.MaxPool1d(2)
            self.flatten = nn.Flatten()
            with torch.no_grad():
                dummy = torch.zeros(1, 1, num_features)
                x = self.pool1(torch.relu(self.bn1(self.conv1(dummy))))
                x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
                flat_size = x.numel()
            self.fc1 = nn.Linear(flat_size, 128)
            self.dropout1 = nn.Dropout(0.5)
            self.fc2 = nn.Linear(128, num_classes)
        def forward(self, x):
            x = torch.relu(self.bn1(self.conv1(x)))
            x = self.pool1(x)
            x = torch.relu(self.bn2(self.conv2(x)))
            x = self.pool2(x)
            x = self.flatten(x)
            x = torch.relu(self.fc1(x))
            x = self.dropout1(x)
            x = self.fc2(x)
            return x
    return SimpleCNN()

# Prepare tensors
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch_X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
torch_X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
torch_y_train = torch.tensor(y_train, dtype=torch.long).to(device)
torch_y_val = torch.tensor(y_val, dtype=torch.long).to(device)

num_features = X_cnn.shape[2]
num_classes = len(np.unique(y))

best_acc = 0
best_params = None

for lr in learning_rates:
    for batch_size in batch_sizes:
        for num_filters in num_filters_list:
            print(f"\nTrying lr={lr}, batch_size={batch_size}, num_filters={num_filters}")
            model = make_cnn(num_features, num_classes, num_filters).to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
            optimizer = optim.Adam(model.parameters(), lr=lr)
            train_ds = TensorDataset(torch_X_train, torch_y_train)
            val_ds = TensorDataset(torch_X_val, torch_y_val)
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=batch_size)
            # Train for a few epochs with progress bar
            for epoch in range(10):
                model.train()
                for xb, yb in tqdm(train_loader, desc=f"lr={lr}, bs={batch_size}, nf={num_filters} Epoch {epoch+1}/10"):
                    xb, yb = xb.to(device), yb.to(device)
                    optimizer.zero_grad()
                    out = model(xb)
                    loss = criterion(out, yb)
                    loss.backward()
                    optimizer.step()
            # Evaluate
            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    out = model(xb)
                    preds = torch.argmax(out, dim=1)
                    val_correct += (preds == yb).sum().item()
                    val_total += yb.size(0)
            val_acc = val_correct / val_total
            print(f"Validation accuracy: {val_acc:.4f}")
            if val_acc > best_acc:
                best_acc = val_acc
                best_params = {'lr': lr, 'batch_size': batch_size, 'num_filters': num_filters}

print(f"\nBest validation accuracy: {best_acc:.4f}")
print(f"Best hyperparameters: {best_params}")
