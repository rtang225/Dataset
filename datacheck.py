import pandas as pd
import numpy as np
targets = np.load('week_targets.npy', allow_pickle=True)
bins = [0, 0.1, 1, 10, float('inf')]
# bins = [0, 10, 100, 1000, float('inf')]
# bins = [0, 10, 500, float('inf')]
print("Targets shape:", targets.shape)
labels = list(range(len(bins)))
target_classes = np.digitize(targets, bins, right=False)
# Count the number of samples in each class
binned = pd.cut(targets, bins=bins, right=False, labels=False)
class_counts = pd.Series(binned).value_counts().sort_index()
print("Class counts:")
for label, count in class_counts.items():
    print(f"Class {label}: {count} samples")