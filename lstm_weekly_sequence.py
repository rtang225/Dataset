import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix, accuracy_score
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence

sequences = np.load('week_sequences.npy', allow_pickle=True)
targets = np.log10(np.load('week_targets.npy', allow_pickle=True)+1)
print('Data Loaded!')

# Replace NaN values in sequences and targets with 0.0
sequences = [np.nan_to_num(s, nan=0.0) for s in sequences]
targets = np.nan_to_num(targets, nan=0.0)

# Train/test split
indices = np.arange(len(sequences))
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)

seq_tensors = [torch.tensor(s, dtype=torch.float32) for s in sequences]
train_seqs = [seq_tensors[i] for i in train_idx]
test_seqs = [seq_tensors[i] for i in test_idx]
train_targets = torch.tensor(targets[train_idx], dtype=torch.float32)
test_targets = torch.tensor(targets[test_idx], dtype=torch.float32)

train_seqs_padded = pad_sequence(train_seqs, batch_first=True)
test_seqs_padded = pad_sequence(test_seqs, batch_first=True)

# Custom Dataset
class WeekSequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = WeekSequenceDataset(train_seqs_padded, train_targets)
test_dataset = WeekSequenceDataset(test_seqs_padded, test_targets)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# LSTM Model
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=32, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Use last output
        out = self.fc(out)
        return out.squeeze(1)

# Example usage
input_size = train_seqs_padded.shape[2]
model = SimpleLSTM(input_size)

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Move data and model to device
train_seqs_padded = train_seqs_padded.to(device)
test_seqs_padded = test_seqs_padded.to(device)
train_targets = train_targets.to(device)
test_targets = test_targets.to(device)
model = model.to(device)

# Training setup
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)
num_epochs = 20
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        xb = xb.to(device)
        yb = yb.to(device)
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
            xb = xb.to(device)
            yb = yb.to(device)
            out = model(xb)
            loss = criterion(out, yb)
            val_running_loss += loss.item() * xb.size(0)
    val_loss = val_running_loss / len(test_loader.dataset)
    val_losses.append(val_loss)
    print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

# Evaluation
model.eval()
preds = []
trues = []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        yb = yb.to(device)
        out = model(xb)
        preds.append(out.cpu().numpy())
        trues.append(yb.cpu().numpy())
preds = np.concatenate(preds)
trues = np.concatenate(trues)
preds_rounded = np.floor(preds)
preds_rounded = np.clip(preds_rounded, 0, 3)
trues_rounded = np.floor(trues)
trues_rounded = np.clip(trues_rounded, 0, 3)

# Final evaluation metrics
mse = mean_squared_error(trues, preds)
r2 = r2_score(trues, preds)
print(f"Final MSE: {mse:.4f}")
print(f"Final R^2: {r2:.4f}")

print('Classification Report (Rounded):')
print(classification_report(trues_rounded, preds_rounded, digits=3))
print('Confusion Matrix (Rounded):')
print(confusion_matrix(trues_rounded, preds_rounded))
acc = accuracy_score(trues_rounded, preds_rounded)
print(f'Accuracy (Rounded): {acc:.3f}')
# List the number of data for each class in the final validation test
unique, counts = np.unique(trues_rounded, return_counts=True)
total = len(trues_rounded)
print('Number of samples per class in y_true_rounded:')
for u, c in zip(unique, counts):
    percent = 100 * c / total
    print(f'Class {int(u)}: {c} ({percent:.2f}%)')

# Plot loss curve
plt.plot(train_losses)
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('LSTM Training Loss')
plt.show()
