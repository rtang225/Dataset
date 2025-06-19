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

# Reshape for RNN: (samples, timesteps, features)
X_rnn = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_rnn, y, test_size=0.2, random_state=42)

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

# Define RNN model
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, _ = self.rnn(x)
        out = out[:, -1, :]  # Take last time step
        out = self.fc(out)
        return out

num_classes = len(np.unique(y))
input_size = 1
hidden_size = 64
model = SimpleRNN(input_size, hidden_size, num_classes).to(device)
print(f"Model is on device: {next(model.parameters()).device}")

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 30
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
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
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")

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
