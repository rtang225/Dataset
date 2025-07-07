import pandas as pd
import numpy as np
targets = np.log10(np.load('week_targets.npy', allow_pickle=True)+1)
bins = [0, 0.1, 1, 10, float('inf')]
print("Targets shape:", targets.shape)
labels = list(range(len(bins)-1))
target_classes = np.digitize(targets, bins, right=False) - 1
# Count the number of samples in each class
class_counts = pd.Series(target_classes).value_counts().sort_index()
print("Class counts:")
for label, count in class_counts.items():
    print(f"Class {label}: {count} samples")