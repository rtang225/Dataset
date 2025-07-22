import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from transformers import get_cosine_schedule_with_warmup

sequences = np.load('week_sequences.npy', allow_pickle=True)
targets = np.log10(np.load('week_targets.npy', allow_pickle=True)+0.01)
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
num_samples = len(sequences)
indices = np.arange(num_samples)
np.random.seed(42)
np.random.shuffle(indices)
split = int(num_samples * 0.8)
train_idx, test_idx = indices[:split], indices[split:]
seq_tensors = [torch.tensor(s, dtype=torch.float32) for s in sequences]
train_seqs = [seq_tensors[i] for i in train_idx]
test_seqs = [seq_tensors[i] for i in test_idx]
train_targets = torch.tensor(targets[train_idx], dtype=torch.float32)
test_targets = torch.tensor(targets[test_idx], dtype=torch.float32)

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

class SimpleTransformerRegressor(nn.Module):
    def __init__(self, input_size, seq_len, d_model=128, nhead=4, num_layers=3, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.ln_input = nn.LayerNorm(d_model)
        self.positional_encoding = nn.Parameter(self._get_positional_encoding(seq_len, d_model), requires_grad=False)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.ModuleList([
            nn.Sequential(encoder_layer, nn.LayerNorm(d_model)) for _ in range(num_layers)
        ])
        self.fc = nn.Linear(d_model, 1)
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
            x = x * mask.float()
            pooled = x.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            pooled = x.mean(dim=1)
        out = self.fc(pooled).squeeze(-1)
        return out

input_size = train_seqs[0].shape[1]
seq_len = max([x.shape[0] for x in train_seqs])
model = SimpleTransformerRegressor(input_size, seq_len)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
train_targets = train_targets.to(device)
test_targets = test_targets.to(device)
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=1e-3)
num_epochs = 50
warmup_steps = 1000
scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, num_epochs)
train_losses = []
val_losses = []
aux_train_loss = []
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    aux_loss_sum = 0.0
    for xb, yb, mask in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        xb = xb.to(device)
        yb = yb.to(device)
        mask = mask.to(device)
        optimizer.zero_grad()
        out = model(xb, mask)
        # Main loss: regression to target
        main_loss = F.mse_loss(out, yb)
        # Auxiliary loss: predict sequence mean
        seq_mean = xb.mean(dim=1).mean(dim=1)
        aux_loss = F.mse_loss(out, seq_mean)
        total_loss = main_loss + 0.1 * aux_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += total_loss.item() * xb.size(0)
        aux_loss_sum += aux_loss.item() * xb.size(0)
    train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(train_loss)
    aux_train_loss.append(aux_loss_sum / len(train_loader.dataset))
    model.eval()
    val_running_loss = 0.0
    val_aux_loss_sum = 0.0
    all_val_preds = []
    all_val_trues = []
    for xb, yb, mask in tqdm(test_loader, desc=f"Val {epoch+1}/{num_epochs}"):
        xb = xb.to(device)
        yb = yb.to(device)
        mask = mask.to(device)
        out = model(xb, mask)
        main_loss = F.mse_loss(out, yb)
        seq_mean = xb.mean(dim=1).mean(dim=1)
        aux_loss = F.mse_loss(out, seq_mean)
        val_running_loss += (main_loss + 0.1 * aux_loss).item() * xb.size(0)
        val_aux_loss_sum += aux_loss.item() * xb.size(0)
        all_val_preds.append(out.detach().cpu().numpy())
        all_val_trues.append(yb.cpu().numpy())
    val_loss = val_running_loss / len(test_loader.dataset)
    val_losses.append(val_loss)
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Aux Train Loss: {aux_train_loss[-1]:.4f}")
all_val_preds = np.concatenate(all_val_preds)
all_val_trues = np.concatenate(all_val_trues)
plt.figure(figsize=(8,5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.plot(aux_train_loss, label='Auxiliary Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Transformer Regression Training and Validation Loss')
plt.legend()
plt.tight_layout()
plt.show()
plt.figure(figsize=(8,5))
plt.scatter(all_val_trues, all_val_preds, alpha=0.5)
plt.xlabel('True Target')
plt.ylabel('Predicted Target')
plt.title('True vs Predicted Regression Targets')
plt.tight_layout()
plt.show()
plt.figure(figsize=(8,5))
plt.hist(all_val_preds - all_val_trues, bins=50, alpha=0.7)
plt.xlabel('Prediction Error')
plt.ylabel('Count')
plt.title('Prediction Error Distribution')
plt.tight_layout()
plt.show()
