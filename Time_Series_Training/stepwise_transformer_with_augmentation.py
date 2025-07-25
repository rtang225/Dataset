import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support, f1_score
from tqdm import tqdm
from sklearn.model_selection import StratifiedShuffleSplit
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
import torch.nn.functional as F
from transformers import get_cosine_schedule_with_warmup
import matplotlib.pyplot as plt
import seaborn as sns

# Load sequences and targets
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

# Stepwise bins
bins = [0, 10, float('inf')]
labels = list(range(len(bins)-1))
target_classes = np.digitize(targets, bins, right=False) - 1

# Data augmentation function
np.random.seed(42)
def augment_sequence(seq):
    noise = np.random.normal(0, 0.01, seq.shape)
    seq_noisy = seq + noise
    scale = np.random.uniform(0.95, 1.05)
    seq_scaled = seq_noisy * scale
    shift = np.random.randint(-5, 6)
    seq_shifted = np.roll(seq_scaled, shift, axis=0)
    dropout_mask = np.random.binomial(1, 0.98, seq_shifted.shape)
    seq_dropout = seq_shifted * dropout_mask
    drift = np.cumsum(np.random.normal(0, 0.001, seq_dropout.shape[0]))
    seq_drifted = seq_dropout + drift[:, None]
    warp = np.random.uniform(0.9, 1.1, seq_drifted.shape[0])
    seq_warped = seq_drifted * warp[:, None]
    if seq_warped.shape[1] > 1:
        channel_dropout = np.random.binomial(1, 0.95, seq_warped.shape[1])
        seq_channel_dropout = seq_warped * channel_dropout[None, :]
    else:
        seq_channel_dropout = seq_warped
    n_spikes = np.random.randint(1, 4)
    for _ in range(n_spikes):
        spike_t = np.random.randint(0, seq_channel_dropout.shape[0])
        spike_f = np.random.randint(0, seq_channel_dropout.shape[1])
        seq_channel_dropout[spike_t, spike_f] += np.random.uniform(0.5, 1.5)
    return seq_channel_dropout

# Step 1: Binary classification (0-10 vs 10-inf)
binary_bins = [0, 10, float('inf')]
binary_classes = np.digitize(targets, binary_bins, right=False) - 1
keep_indices = np.where(np.isin(binary_classes, [0, 1]))[0]
binary_sequences = [sequences[i] for i in keep_indices]
binary_targets = targets[keep_indices]
binary_class_labels = binary_classes[keep_indices]
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(sss.split(np.arange(len(binary_sequences)), binary_class_labels))
seq_tensors = [torch.tensor(s, dtype=torch.float32) for s in binary_sequences]
train_seqs = [seq_tensors[i] for i in train_idx]
test_seqs = [seq_tensors[i] for i in test_idx]
train_targets = torch.tensor(binary_class_labels[train_idx], dtype=torch.long)
test_targets = torch.tensor(binary_class_labels[test_idx], dtype=torch.long)

# Oversample training set
train_target_classes = train_targets.cpu().numpy() if torch.is_tensor(train_targets) else np.array(train_targets)
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
train_seqs = [train_seqs[i] for i in all_train_indices]
train_targets = train_targets[all_train_indices] if torch.is_tensor(train_targets) else np.array(train_targets)[all_train_indices]

# Augment training sequences
train_seqs = [torch.tensor(augment_sequence(s.numpy()), dtype=torch.float32) for s in train_seqs]

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
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=collate_fn)

class SimpleTransformerClassifier(nn.Module):
    def __init__(self, input_size, num_classes, seq_len, d_model=128, nhead=4, num_layers=2, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.ln_input = nn.LayerNorm(d_model)
        self.positional_encoding = nn.Parameter(self._get_positional_encoding(seq_len, d_model), requires_grad=False)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.ModuleList([
            nn.Sequential(encoder_layer, nn.LayerNorm(d_model)) for _ in range(num_layers)
        ])
        self.attn_pool = nn.Linear(d_model, 1)
        self.fc = nn.Linear(d_model, 2)
        self.seq_len = seq_len
    def _get_positional_encoding(self, seq_len, d_model):
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
    def forward(self, x, mask=None):
        x = self.input_proj(x)
        x = self.ln_input(x)
        x = x + self.positional_encoding[:, :x.size(1), :].to(x.device)
        for block in self.transformer_encoder:
            x = block(x)
        if mask is not None:
            mask = mask.unsqueeze(-1)
            attn_weights = torch.softmax(self.attn_pool(x).masked_fill(~mask, float('-inf')), dim=1)
        else:
            attn_weights = torch.softmax(self.attn_pool(x), dim=1)
        pooled = torch.sum(attn_weights * x, dim=1)
        out = self.fc(pooled)
        return out, attn_weights.detach().cpu().numpy()

input_size = train_seqs[0].shape[1]
seq_len = max([x.shape[0] for x in train_seqs])
model = SimpleTransformerClassifier(input_size, 2, seq_len)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_targets = train_targets.to(device)
test_targets = test_targets.to(device)
model = model.to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-3)
num_epochs = 50
warmup_steps = 750
scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, num_epochs)
train_losses = []
val_losses = []
macro_f1_score = []
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for xb, yb, mask in tqdm(train_loader, desc=f"Binary Epoch {epoch+1}/{num_epochs}"):
        xb = xb.to(device)
        yb = yb.to(device)
        mask = mask.to(device)
        optimizer.zero_grad()
        out, _ = model(xb, mask)
        loss = criterion(out, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item() * xb.size(0)
    if epoch < warmup_steps:
        for g in optimizer.param_groups:
            g['lr'] = 0.0001 + (0.001 - 0.0001) * (epoch + 1) / warmup_steps
    else:
        scheduler.step()
    train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(train_loss)
    model.eval()
    val_running_loss = 0.0
    all_val_preds = []
    all_val_trues = []
    for xb, yb, mask in tqdm(test_loader, desc=f"Binary Val {epoch+1}/{num_epochs}"):
        xb = xb.to(device)
        yb = yb.to(device)
        mask = mask.to(device)
        out, _ = model(xb, mask)
        loss = criterion(out, yb)
        val_running_loss += loss.item() * xb.size(0)
        preds = torch.argmax(out, dim=1)
        all_val_preds.append(preds.cpu().numpy())
        all_val_trues.append(yb.cpu().numpy())
    val_loss = val_running_loss / len(test_loader.dataset)
    val_losses.append(val_loss)
    all_val_preds = np.concatenate(all_val_preds)
    all_val_trues = np.concatenate(all_val_trues)
    macro_f1 = f1_score(all_val_trues, all_val_preds, average='macro')
    macro_f1_score.append(macro_f1)
    print(f"[Binary] Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Macro F1: {macro_f1:.4f}")
print("\nBinary classification metrics:")
print(classification_report(all_val_trues, all_val_preds, digits=3))
print('Confusion Matrix:')
cm = confusion_matrix(all_val_trues, all_val_preds)
print(cm)
acc = accuracy_score(all_val_trues, all_val_preds)
print(f'Accuracy: {acc:.3f}')
macro_f1 = f1_score(all_val_trues, all_val_preds, average='macro')
print(f'Macro F1: {macro_f1:.3f}\n')

# Step 2: Multiclass for 10-inf
binary_val_preds_saved = all_val_preds.copy()
binary_test_indices_saved = np.array(test_idx)
selected_indices = binary_test_indices_saved[binary_val_preds_saved == 1]
selected_sequences = [sequences[i] for i in selected_indices]
selected_targets = targets[selected_indices]
multi_bins = [10, 100, 1000, float('inf')]
multi_classes = np.digitize(selected_targets, multi_bins, right=False) - 1
keep_indices = np.where(np.isin(multi_classes, [0, 1, 2]))[0]
multi_sequences = [selected_sequences[i] for i in keep_indices]
multi_targets = selected_targets[keep_indices]
multi_class_labels = multi_classes[keep_indices]
if len(multi_sequences) == 0:
    print('No samples predicted as 10-inf in binary step. Skipping multi-class step.')
else:
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(sss.split(np.arange(len(multi_sequences)), multi_class_labels))
    seq_tensors = [torch.tensor(s, dtype=torch.float32) for s in multi_sequences]
    train_seqs = [seq_tensors[i] for i in train_idx]
    test_seqs = [seq_tensors[i] for i in test_idx]
    train_targets = torch.tensor(multi_class_labels[train_idx], dtype=torch.long)
    test_targets = torch.tensor(multi_class_labels[test_idx], dtype=torch.long)
    # Oversample and augment
    train_target_classes = train_targets.cpu().numpy() if torch.is_tensor(train_targets) else np.array(train_targets)
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
    train_seqs = [train_seqs[i] for i in all_train_indices]
    train_targets = train_targets[all_train_indices] if torch.is_tensor(train_targets) else np.array(train_targets)[all_train_indices]
    train_seqs = [torch.tensor(augment_sequence(s.numpy()), dtype=torch.float32) for s in train_seqs]
    train_dataset = WeekSequenceDataset(train_seqs, train_targets)
    test_dataset = WeekSequenceDataset(test_seqs, test_targets)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=collate_fn)
    model = SimpleTransformerClassifier(input_size, 3, seq_len)
    train_targets = train_targets.to(device)
    test_targets = test_targets.to(device)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-3)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, num_epochs)
    train_losses = []
    val_losses = []
    macro_f1_score = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for xb, yb, mask in tqdm(train_loader, desc=f"Multi Epoch {epoch+1}/{num_epochs}"):
            xb = xb.to(device)
            yb = yb.to(device)
            mask = mask.to(device)
            optimizer.zero_grad()
            out, _ = model(xb, mask)
            loss = criterion(out, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
        if epoch < warmup_steps:
            for g in optimizer.param_groups:
                g['lr'] = 0.0001 + (0.001 - 0.0001) * (epoch + 1) / warmup_steps
        else:
            scheduler.step()
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        model.eval()
        val_running_loss = 0.0
        all_val_preds = []
        all_val_trues = []
        for xb, yb, mask in tqdm(test_loader, desc=f"Multi Val {epoch+1}/{num_epochs}"):
            xb = xb.to(device)
            yb = yb.to(device)
            mask = mask.to(device)
            out, _ = model(xb, mask)
            loss = criterion(out, yb)
            val_running_loss += loss.item() * xb.size(0)
            preds = torch.argmax(out, dim=1)
            all_val_preds.append(preds.cpu().numpy())
            all_val_trues.append(yb.cpu().numpy())
        val_loss = val_running_loss / len(test_loader.dataset)
        val_losses.append(val_loss)
        all_val_preds = np.concatenate(all_val_preds)
        all_val_trues = np.concatenate(all_val_trues)
        macro_f1 = f1_score(all_val_trues, all_val_preds, average='macro')
        macro_f1_score.append(macro_f1)
        print(f"[Multi] Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Macro F1: {macro_f1:.4f}")
    print("\nMulti-class classification metrics (10-inf):")
    print(classification_report(all_val_trues, all_val_preds, digits=3))
    print('Confusion Matrix:')
    cm = confusion_matrix(all_val_trues, all_val_preds)
    print(cm)
    acc = accuracy_score(all_val_trues, all_val_preds)
    print(f'Accuracy: {acc:.3f}')
    macro_f1 = f1_score(all_val_trues, all_val_preds, average='macro')
    print(f'Macro F1: {macro_f1:.3f}\n')
print('Stepwise binary and multi-class training and evaluation complete.')
