import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support, f1_score
from tqdm import tqdm
from sklearn.model_selection import StratifiedShuffleSplit
from torch.nn.utils.rnn import pad_sequence
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
import seaborn as sns
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
from transformers import get_cosine_schedule_with_warmup

# Load sequences and targets
sequences = np.load('week_sequences.npy', allow_pickle=True)
targets = np.load('week_targets.npy', allow_pickle=True)
sequences = [np.nan_to_num(s, nan=0.0) for s in sequences]
targets = np.nan_to_num(targets, nan=0.0)
remove = []

# Remove sequences longer than 168 timesteps
for i in range(len(sequences)):
    if len(sequences[i]) > 168:
        remove.insert(0, i)
for i in range(len(remove)):
    idx = remove[i]
    sequences.pop(idx)
    targets = np.delete(targets, idx)

bins = [0, 0.1, 1, 10, float('inf')]
# bins = [0, 1, 10, 100, float('inf')]
# bins = [0, 10, 250, float('inf')]
labels = list(range(len(bins)-1))
target_classes = np.digitize(targets, bins, right=False) - 1

# Stratified train/val split (no oversampling)
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(sss.split(np.arange(len(sequences)), target_classes))

seq_tensors = [torch.tensor(s, dtype=torch.float32) for s in sequences]
train_seqs = [seq_tensors[i] for i in train_idx]
test_seqs = [seq_tensors[i] for i in test_idx]
train_targets = torch.tensor(target_classes[train_idx], dtype=torch.long)
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
        self.fc = nn.Linear(d_model, num_classes)
        self.seq_len = seq_len
    def _get_positional_encoding(self, seq_len, d_model):
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # shape (1, seq_len, d_model)
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
num_classes = len(labels)
model = SimpleTransformerClassifier(input_size, num_classes, seq_len)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
train_targets = train_targets.to(device)
test_targets = test_targets.to(device)
model = model.to(device)

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # Can be a float or a tensor/list of per-class weights
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        if self.alpha is not None:
            if isinstance(self.alpha, (list, torch.Tensor)):
                if not isinstance(self.alpha, torch.Tensor):
                    alpha = torch.tensor(self.alpha, dtype=inputs.dtype, device=inputs.device)
                else:
                    alpha = self.alpha.to(inputs.device)
                at = alpha[targets]
            else:
                at = self.alpha
            focal_loss = at * (1 - pt) ** self.gamma * ce_loss
        else:
            focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# Manual class weights (tunable)
# manual_weights = torch.tensor([0.69, 0.925, 1.26, 1.13], dtype=torch.float32).to(device) # 0.1, 1, 10, 100
# manual_weights = torch.tensor([0.5, 1, 1.5, 1], dtype=torch.float32).to(device) # 1, 10, 100, 1000

# criterion = nn.CrossEntropyLoss(weight=manual_weights, label_smoothing=0.1)
# criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
criterion = FocalLoss(alpha=[0.8, 1, 1.2, 1], gamma=3, reduction='mean')
# criterion = FocalLoss(alpha=None, gamma=3, reduction='mean')
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-3)
# scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)
num_epochs = 150
warmup_steps = 750
scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, num_epochs)

train_losses = []
val_losses = []
per_class_f1 = []
macro_f1_score = []
aux_train_class_loss = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    class_loss_sum = np.zeros(num_classes)
    class_loss_count = np.zeros(num_classes)
    aux_loss_sum = 0.0
    for xb, yb, mask in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        xb = xb.to(device)
        yb = yb.to(device)
        mask = mask.to(device)
        optimizer.zero_grad()
        out, _ = model(xb, mask)
        ce_loss = F.cross_entropy(out, yb, reduction='none')
        pt = torch.exp(-ce_loss)
        if criterion.alpha is not None:
            if isinstance(criterion.alpha, (list, torch.Tensor)):
                alpha = torch.tensor(criterion.alpha, dtype=out.dtype, device=out.device) if not isinstance(criterion.alpha, torch.Tensor) else criterion.alpha.to(out.device)
                at = alpha[yb]
            else:
                at = criterion.alpha
            focal_loss = at * (1 - pt) ** criterion.gamma * ce_loss
        else:
            focal_loss = (1 - pt) ** criterion.gamma * ce_loss
        # Auxiliary task: predict sequence mean as regression
        seq_mean = xb.mean(dim=1).mean(dim=1)  # shape [batch_size]
        aux_pred = out.mean(dim=1)  # shape [batch_size]
        aux_loss = F.mse_loss(aux_pred, seq_mean)
        aux_loss_sum += aux_loss.item() * xb.size(0)
        for c in range(num_classes):
            mask_c = (yb == c)
            if mask_c.any():
                class_loss_sum[c] += focal_loss[mask_c].sum().item()
                class_loss_count[c] += mask_c.sum().item()
        loss = focal_loss.mean() + 0.1 * aux_loss  # Weighted sum
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item() * xb.size(0)
    avg_class_loss = class_loss_sum / np.maximum(class_loss_count, 1)
    aux_train_class_loss.append(avg_class_loss)
    print(f"Per-class train loss: " + ", ".join([f"Class {i}: {avg_class_loss[i]:.4f}" for i in range(num_classes)]))
    print(f"Auxiliary regression loss: {aux_loss_sum / len(train_loader.dataset):.4f}")
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
    all_val_attn = []
    for xb, yb, mask in tqdm(test_loader, desc=f"Val {epoch+1}/{num_epochs}"):
        xb = xb.to(device)
        yb = yb.to(device)
        mask = mask.to(device)
        out, attn_weights = model(xb, mask)
        loss = criterion(out, yb)
        val_running_loss += loss.item() * xb.size(0)
        preds = torch.argmax(out, dim=1)
        all_val_preds.append(preds.cpu().numpy())
        all_val_trues.append(yb.cpu().numpy())
        all_val_attn.append(attn_weights)
    val_loss = val_running_loss / len(test_loader.dataset)
    val_losses.append(val_loss)
    all_val_preds = np.concatenate(all_val_preds)
    all_val_trues = np.concatenate(all_val_trues)
    precision, recall, f1, _ = precision_recall_fscore_support(all_val_trues, all_val_preds, labels=np.arange(num_classes), zero_division=0)
    macro_f1 = f1_score(all_val_trues, all_val_preds, average='macro')
    macro_f1_score.append(macro_f1)
    per_class_f1.append(f1)
    print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Macro F1: {macro_f1:.4f}")
    for i in range(num_classes):
        if epoch == 0 or (epoch+1)%5 == 0:
            print(f"Class {i}: Precision={precision[i]:.3f}, Recall={recall[i]:.3f}, F1={f1[i]:.3f}")

model.eval()
all_preds = []
all_trues = []
all_attn = []
with torch.no_grad():
    for xb, yb, mask in test_loader:
        xb = xb.to(device)
        yb = yb.to(device)
        mask = mask.to(device)
        out, attn_weights = model(xb, mask)
        preds = torch.argmax(out, dim=1)
        all_preds.append(preds.cpu().numpy())
        all_trues.append(yb.cpu().numpy())
        all_attn.append(attn_weights)
all_preds = np.concatenate(all_preds)
all_trues = np.concatenate(all_trues)
all_attn = np.concatenate(all_attn, axis=0)

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

"""plt.figure(figsize=(8,5))
plt.plot(train_losses, label='Train Loss', marker='o')
plt.plot(val_losses, label='Validation Loss', marker='o')
plt.plot(macro_f1_score, label='Macro F1 Score', marker='o')
plt.xlabel('Epoch')
plt.ylabel('CrossEntropy Loss / Macro F1')
plt.title('Transformer Classification Training, Validation Loss, and Macro F1')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()"""

aux_train_class_loss = np.array(aux_train_class_loss)
plt.figure(figsize=(8,5))
for i in range(num_classes):
    plt.plot(aux_train_class_loss[:, i], label=f'Class {i} Train Loss', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Per-Class Train Loss')
plt.title('Per-Class Train Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()