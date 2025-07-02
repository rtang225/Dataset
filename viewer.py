import numpy as np
data = np.load('week_sequences.npy', allow_pickle=True)
for seq in data:
    if np.isnan(seq).any():
        for i, val in enumerate(seq):
            if np.isnan(val).any():
                print(f"NaN found in sequence at index {i}: {val}")
        print("NaN found in sequence:", seq)