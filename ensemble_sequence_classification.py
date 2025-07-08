import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load predictions (assume saved as .npy files from each model)
lstm_preds = np.load('lstm_preds.npy')
gru_preds = np.load('gru_preds.npy')
transformer_preds = np.load('transformer_preds.npy')
# Load true labels
all_trues = np.load('all_trues.npy')

# Weighted average ensemble (example weights, adjust as needed)
lstm_weight = 0.4
gru_weight = 0.3
transformer_weight = 0.3

# Stack predictions as probabilities (if available), else as class indices
# If you have probabilities, use them directly. If not, convert class indices to one-hot
num_classes = int(max(lstm_preds.max(), gru_preds.max(), transformer_preds.max()) + 1)
lstm_onehot = np.eye(num_classes)[lstm_preds.astype(int)]
gru_onehot = np.eye(num_classes)[gru_preds.astype(int)]
transformer_onehot = np.eye(num_classes)[transformer_preds.astype(int)]

ensemble_probs = (
    lstm_weight * lstm_onehot +
    gru_weight * gru_onehot +
    transformer_weight * transformer_onehot
)
ensemble_preds = np.argmax(ensemble_probs, axis=1)

print('Ensemble Classification Report:')
print(classification_report(all_trues, ensemble_preds, digits=3))
print('Ensemble Confusion Matrix:')
print(confusion_matrix(all_trues, ensemble_preds))
acc = accuracy_score(all_trues, ensemble_preds)
print(f'Ensemble Accuracy: {acc:.3f}')
