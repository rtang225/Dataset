import numpy as np

# Load both .npy files
seq1 = np.load('week_targets.npy', allow_pickle=True)
seq2 = np.load('week_targets2.npy', allow_pickle=True)

# Combine sequences
combined = np.concatenate([seq1, seq2], axis=0)

# Save combined file
np.save('week_targets_combined.npy', combined)
print('Saved combined sequences to week_targets_combined.npy')
