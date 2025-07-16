import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support, f1_score
import seaborn as sns

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
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(sss.split(np.arange(len(sequences)), target_classes))
train_target_classes = target_classes[train_idx]
class_counts = Counter(train_target_classes)
max_count = max(class_counts.values())
indices_by_class = {c: np.where(train_target_classes == c)[0] for c in class_counts}
all_train_indices = []
for c, idxs in indices_by_class.items():
    n_to_add = max_count - len(idxs)
    if n_to_add > 0:
        idxs_oversampled = np.random.choice(idxs, n_to_add, replace=True)
        all_train_indices.extend(idxs.tolist() + idxs_oversampled.tolist())
    else:
        all_train_indices.extend(idxs.tolist())
all_train_indices = np.array(all_train_indices)
np.random.shuffle(all_train_indices)
oversampled_train_idx = train_idx[all_train_indices]
seq_tensors = [torch.tensor(s, dtype=torch.float32) for s in sequences]
train_seqs = [seq_tensors[i] for i in oversampled_train_idx]
test_seqs = [seq_tensors[i] for i in test_idx]
train_targets = torch.tensor(target_classes[oversampled_train_idx], dtype=torch.long)
test_targets = torch.tensor(target_classes[test_idx], dtype=torch.long)

def collate_fn(batch):
    X, y = zip(*batch)
    X_padded = pad_sequence(X, batch_first=True)
    lengths = torch.tensor([x.shape[0] for x in X])
    mask = torch.arange(X_padded.shape[1])[None, :] < lengths[:, None]
    return X_padded, torch.tensor(y), mask

class WeekSequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = WeekSequenceDataset(train_seqs, train_targets)
test_dataset = WeekSequenceDataset(test_seqs, test_targets)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn)

class SimpleLSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
    def forward(self, x, mask=None):
        out, _ = self.lstm(x)
        if mask is not None:
            mask = mask.unsqueeze(-1)
            out = out * mask.float()
            pooled = out.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            pooled = out.mean(dim=1)
        out = self.fc(pooled)
        return out

input_size = train_seqs[0].shape[1]
num_classes = len(labels)
model = SimpleLSTMClassifier(input_size, hidden_size=64, num_layers=2, num_classes=num_classes, dropout=0.2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
train_targets = train_targets.to(device)
test_targets = test_targets.to(device)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-3)
num_epochs = 30
train_losses = []
val_losses = []
per_class_f1 = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for xb, yb, mask in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)
        mask = mask.to(device)
        optimizer.zero_grad()
        out = model(xb, mask)
        loss = criterion(out, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item() * xb.size(0)
    train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(train_loss)
    # Validation
    model.eval()
    val_running_loss = 0.0
    all_val_preds = []
    all_val_trues = []
    with torch.no_grad():
        for xb, yb, mask in test_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            mask = mask.to(device)
            out = model(xb, mask)
            loss = criterion(out, yb)
            val_running_loss += loss.item() * xb.size(0)
            preds = torch.argmax(out, dim=1)
            all_val_preds.append(preds.cpu().numpy())
            all_val_trues.append(yb.cpu().numpy())
    val_loss = val_running_loss / len(test_loader.dataset)
    val_losses.append(val_loss)
    all_val_preds = np.concatenate(all_val_preds)
    all_val_trues = np.concatenate(all_val_trues)
    precision, recall, f1, _ = precision_recall_fscore_support(all_val_trues, all_val_preds, labels=np.arange(num_classes), zero_division=0)
    macro_f1 = f1_score(all_val_trues, all_val_preds, average='macro')
    per_class_f1.append(f1)
    print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Macro F1: {macro_f1:.4f}")
    for i in range(num_classes):
        print(f"Class {i}: Precision={precision[i]:.3f}, Recall={recall[i]:.3f}, F1={f1[i]:.3f}")

# Final evaluation
model.eval()
all_preds = []
all_trues = []
with torch.no_grad():
    for xb, yb, mask in test_loader:
        xb = xb.to(device)
        yb = yb.to(device)
        mask = mask.to(device)
        out = model(xb, mask)
        preds = torch.argmax(out, dim=1)
        all_preds.append(preds.cpu().numpy())
        all_trues.append(yb.cpu().numpy())
all_preds = np.concatenate(all_preds)
all_trues = np.concatenate(all_trues)

print('Classification Report:')
print(classification_report(all_trues, all_preds, digits=3))
print('Confusion Matrix:')
cm = confusion_matrix(all_trues, all_preds)
print(cm)
acc = accuracy_score(all_trues, all_preds)
print(f'Accuracy: {acc:.3f}')
macro_f1 = f1_score(all_trues, all_preds, average='macro')
print(f'Macro F1: {macro_f1:.3f}')

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix Heatmap')
plt.tight_layout()
plt.show()

per_class_f1 = np.array(per_class_f1)
plt.figure(figsize=(8,5))
for i in range(num_classes):
    plt.plot(per_class_f1[:, i], label=f'Class {i} F1')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.title('Per-Class F1 Score Over Epochs')
plt.legend()
plt.tight_layout()
plt.show()
