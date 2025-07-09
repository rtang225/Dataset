import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
import itertools

# Load and preprocess data (same as your transformer script)
sequences = np.load('week_sequences.npy', allow_pickle=True)
targets = np.load('week_targets.npy', allow_pickle=True)
sequences = [np.nan_to_num(s, nan=0.0) for s in sequences]
targets = np.nan_to_num(targets, nan=0.0)
remove = []
for i in range(len(sequences)):
    if len(sequences[i]) > 168:
        remove.insert(0, i)
for i in range(len(remove)):
    idx = remove[i]
    sequences.pop(idx)
    targets = np.delete(targets, idx)
bins = [0, 0.1, 1, 10, float('inf')]
labels = list(range(len(bins)-1))
target_classes = np.digitize(targets, bins, right=False) - 1
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
    def __init__(self, input_size, num_classes, seq_len, d_model=256, nhead=8, num_layers=4, dim_feedforward=256, dropout=0.1):
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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Grid search for transformer parameters
param_grid = {
    'd_model': [64, 128, 256],
    'nhead': [2, 4, 8],
    'num_layers': [2, 4, 6],
    'dim_feedforward': [128, 256, 512],
    'dropout': [0.1, 0.2, 0.3],
    'lr': [0.0001, 0.0002, 0.0005, 0.001],
}
best_acc = 0
best_params = None
results = []

for d_model, nhead, num_layers, dim_feedforward, dropout, lr in itertools.product(
    param_grid['d_model'], param_grid['nhead'], param_grid['num_layers'],
    param_grid['dim_feedforward'], param_grid['dropout'], param_grid['lr']):
    if d_model % nhead != 0:
        continue  # nhead must divide d_model
    model = SimpleTransformerClassifier(input_size, num_classes, seq_len, d_model, nhead, num_layers, dim_feedforward, dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    for epoch in range(5):  # Use fewer epochs for grid search
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
        # Validation
        model.eval()
        with torch.no_grad():
            all_preds = []
            all_trues = []
            for xb, yb in test_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                out = model(xb)
                preds = torch.argmax(out, dim=1)
                all_preds.append(preds.cpu().numpy())
                all_trues.append(yb.cpu().numpy())
            all_preds = np.concatenate(all_preds)
            all_trues = np.concatenate(all_trues)
            acc = accuracy_score(all_trues, all_preds)
    results.append((d_model, nhead, num_layers, dim_feedforward, dropout, lr, acc))
    if acc > best_acc:
        best_acc = acc
        best_params = (d_model, nhead, num_layers, dim_feedforward, dropout, lr)
    print(f"Params: d_model={d_model}, nhead={nhead}, num_layers={num_layers}, dim_feedforward={dim_feedforward}, dropout={dropout}, lr={lr} => Acc={acc:.3f}")
print(f"Best params: d_model={best_params[0]}, nhead={best_params[1]}, num_layers={best_params[2]}, dim_feedforward={best_params[3]}, dropout={best_params[4]}, lr={best_params[5]} => Acc={best_acc:.3f}")
