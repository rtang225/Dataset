import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('initialexport.csv')

# Encode non-numeric columns (except target)
for col in df.columns:
    if df[col].dtype == 'O' and col != 'area':
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# Assume 'area' is the target column (change if needed)
X = df.drop(['area'], axis=1, errors='ignore')
y = np.log10(df['area']+1)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

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
        self.elu1 = nn.ELU()
        self.fc2 = nn.Linear(64, 32)
        self.elu2 = nn.ELU()
        self.fc3 = nn.Linear(32, 16)
        self.elu3 = nn.ELU()
        self.fc4 = nn.Linear(16, 8)
        self.elu4 = nn.ELU()
        self.fc5 = nn.Linear(8, 1)
    def forward(self, x):
        x = self.elu1(self.fc1(x))
        x = self.elu2(self.fc2(x))
        x = self.elu3(self.fc3(x))
        x = self.elu4(self.fc4(x))
        x = self.fc5(x)
        return x

num_features = X_train.shape[1]
model = SimpleRegressor(num_features)

# Define log-cosh loss
class LogCoshLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, y_pred, y_true):
        return torch.mean(torch.log(torch.cosh(y_pred - y_true + 1e-12)))

# Replace criterion with log-cosh loss
criterion = nn.MSELoss()  # Using MSELoss for simplicity, can be replaced with LogCoshLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-3)

num_epochs = 100
train_losses = []
val_losses = []
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
    with torch.no_grad():
        for xb, yb in test_loader:
            out = model(xb)
            loss = criterion(out, yb)
            val_running_loss += loss.item() * xb.size(0)
    val_loss = val_running_loss / len(test_loader.dataset)
    val_losses.append(val_loss)
    if (epoch+1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

# Final evaluation
model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor).cpu().numpy().flatten()
    y_pred_rounded = np.floor(y_pred)
    y_pred_rounded = np.where(y_pred_rounded > 3, 3, y_pred_rounded)
    # y_pred = np.expm1(y_pred)
    # y_true = np.expm1(y_test)
    y_true = y_test
    y_true_rounded = np.floor(y_true)
    y_true_rounded = np.where(y_true_rounded > 3, 3, y_true_rounded)
    print(y_pred, y_true, y_pred_rounded, y_true_rounded)
    # Regression metrics
    mse = mean_squared_error(y_true, y_pred)
    mse_rounded = mean_squared_error(y_true_rounded, y_pred_rounded)
    r2 = r2_score(y_true, y_pred)
    r2_rounded = r2_score(y_true_rounded, y_pred_rounded)
    mae = np.mean(np.abs(y_true - y_pred))
    mae_rounded = np.mean(np.abs(y_true_rounded - y_pred_rounded))
    rmse = np.sqrt(mse)
    rmse_rounded = np.sqrt(mse_rounded)
    print(f'Final Mean Squared Error: {mse:.3f}')
    print(f'Final Root Mean Squared Error: {rmse:.3f}')
    print(f'Final Mean Absolute Error: {mae:.3f}')
    print(f'Final R^2 Score: {r2:.3f}')

    # Classification metrics for rounded values
    print('Classification Report (Rounded):')
    print(classification_report(y_true_rounded, y_pred_rounded, digits=3))
    print('Confusion Matrix (Rounded):')
    print(confusion_matrix(y_true_rounded, y_pred_rounded))
    acc = accuracy_score(y_true_rounded, y_pred_rounded)
    print(f'Accuracy (Rounded): {acc:.3f}')
    # List the number of data for each class in the final validation test
    unique, counts = np.unique(y_true_rounded, return_counts=True)
    total = len(y_true_rounded)
    print('Number of samples per class in y_true_rounded:')
    for u, c in zip(unique, counts):
        percent = 100 * c / total
        print(f'Class {int(u)}: {c} ({percent:.2f}%)')

    # Scatter plot: True vs Predicted
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel('True Area')
    plt.ylabel('Predicted Area')
    plt.title('True vs Predicted Area')
    plt.tight_layout()
    plt.show()

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
