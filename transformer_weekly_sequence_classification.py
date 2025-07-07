import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence

sequences = np.load('week_sequences.npy', allow_pickle=True)
targets = np.load('week_targets.npy', allow_pickle=True)

# Replace NaN values in sequences and targets with 0.0
sequences = [np.nan_to_num(s, nan=0.0) for s in sequences]
targets = np.nan_to_num(targets, nan=0.0)

# Remove sequences longer than 168 timesteps
remove = []
for i in range(len(sequences)):
    if len(sequences[i]) > 168:
        remove.insert(0, i)
for i in range(len(remove)):
    idx = remove[i]
    sequences.pop(idx)
    targets = np.delete(targets, idx)

# Bin targets into classes (example: 4 classes)
bins = [0, 0.1, 1, 10, float('inf')]
labels = list(range(len(bins)-1))
target_classes = np.digitize(targets, bins, right=False) - 1

# Train/test split
indices = np.arange(len(sequences))
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)

seq_tensors = [torch.tensor(s, dtype=torch.float32) for s in sequences]
train_seqs = [seq_tensors[i] for i in train_idx]
test_seqs = [seq_tensors[i] for i in test_idx]
train_targets = torch.tensor(target_classes[train_idx], dtype=torch.long)
test_targets = torch.tensor(target_classes[test_idx], dtype=torch.long)

train_seqs_padded = pad_sequence(train_seqs, batch_first=True)
test_seqs_padded = pad_sequence(test_seqs, batch_first=True)

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
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# Transformer Classifier
class SimpleTransformerClassifier(nn.Module):
    def __init__(self, input_size, num_classes, seq_len, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)
        self.seq_len = seq_len
    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer_encoder(x)
        out = x[:, -1, :]  # Use last token
        out = self.fc(out)
        return out

input_size = train_seqs_padded.shape[2]
seq_len = train_seqs_padded.shape[1]
num_classes = len(labels)
model = SimpleTransformerClassifier(input_size, num_classes, seq_len)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
train_seqs_padded = train_seqs_padded.to(device)
test_seqs_padded = test_seqs_padded.to(device)
train_targets = train_targets.to(device)
test_targets = test_targets.to(device)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0002, weight_decay=1e-3)
num_epochs = 50
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
all_preds = []
all_trues = []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        yb = yb.to(device)
        out = model(xb)
        preds = torch.argmax(out, dim=1)
        all_preds.append(preds.cpu().numpy())
        all_trues.append(yb.cpu().numpy())
all_preds = np.concatenate(all_preds)
all_trues = np.concatenate(all_trues)

print('Classification Report:')
print(classification_report(all_trues, all_preds, digits=3))
print('Confusion Matrix:')
print(confusion_matrix(all_trues, all_preds))
acc = accuracy_score(all_trues, all_preds)
print(f'Accuracy: {acc:.3f}')

# List the number of data for each class in the final validation test
unique, counts = np.unique(all_trues, return_counts=True)
total = len(all_trues)
print('Number of samples per class in y_true_rounded:')
for u, c in zip(unique, counts):
    percent = 100 * c / total
    print(f'Class {int(u)}: {c} ({percent:.2f}%)')

plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('CrossEntropy Loss')
plt.title('Transformer Classification Training and Validation Loss')
plt.legend()
plt.tight_layout()
plt.show()
