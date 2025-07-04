import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence

sequences = np.load('week_sequences.npy', allow_pickle=True)
targets = np.log10(np.load('week_targets.npy', allow_pickle=True)+1)

# Replace NaN values in sequences and targets with 0.0
sequences = [np.nan_to_num(s, nan=0.0) for s in sequences]
targets = np.nan_to_num(targets, nan=0.0)
print(len(sequences))

remove = []
for i in range(len(sequences)):
    if len(sequences[i]) > 168:
        remove.insert(0, i)

for i in range(len(remove)):
    idx = remove[i]
    sequences.pop(idx)
    targets = np.delete(targets, idx)

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

print(f"Train sequences shape: {train_seqs_padded.shape}, Train targets shape: {train_targets.shape}")
print(f"Test sequences shape: {test_seqs_padded.shape}, Test targets shape: {test_targets.shape}")