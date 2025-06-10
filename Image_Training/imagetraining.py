import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight

# Updated image extraction from coordinates
IMAGE_FOLDER = 'images\\wildfire'
def get_image_from_coords(lat, lon):
    filename = f"{lon},{lat}.jpg"
    filepath = os.path.join(IMAGE_FOLDER, filename)
    if os.path.exists(filepath):
        try:
            return Image.open(filepath).convert('RGB')
        except Exception as e:
            # Error is with image -73.15884,46.38819.jpg
            print(f"Error loading image {filepath}: {e}. Returning dummy image.")
            return Image.fromarray(np.uint8(np.random.rand(64, 64, 3) * 255))
    else:
        print(f"Image not found for coordinates ({lat}, {lon}). Returning dummy image.")
        return Image.fromarray(np.uint8(np.random.rand(64, 64, 3) * 255))

# Custom Dataset
default_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

class ImageCoordDataset(Dataset):
    def __init__(self, csv_file, transform=default_transform):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.le = LabelEncoder()
        self.data['Area_Class'] = self.le.fit_transform(self.data['Area_Class'].astype(str))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        lat = self.data.iloc[idx]['Latitude']
        lon = self.data.iloc[idx]['Longtitude']
        label = self.data.iloc[idx]['Area_Class']
        img = get_image_from_coords(lat, lon)
        if self.transform:
            img = self.transform(img)
        return img, label

# Load dataset
csv_path = 'imagelabelsreduced.csv'
dataset = ImageCoordDataset(csv_path)

# Train/test split
indices = np.arange(len(dataset))
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
train_ds = torch.utils.data.Subset(dataset, train_idx)
test_ds = torch.utils.data.Subset(dataset, test_idx)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=64)

# Simple CNN model
def get_num_classes(csv_path):
    df = pd.read_csv(csv_path)
    return len(df['Area_Class'].unique())

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.4)
        self.fc1 = nn.Linear(256 * 4 * 4, 256)
        self.dropout2 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
    def forward(self, x):
        x = self.pool1(torch.sigmoid(self.conv1(x)))
        x = self.pool2(torch.sigmoid(self.conv2(x)))
        x = self.pool3(torch.sigmoid(self.conv3(x)))
        x = self.pool4(torch.sigmoid(self.conv4(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout1(torch.relu(self.fc1(x)))
        x = self.dropout2(torch.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

num_classes = get_num_classes(csv_path)
model = SimpleCNN(num_classes)

# Training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
# Compute class weights for imbalanced classes
y_labels = pd.read_csv(csv_path)['Area_Class']
le = LabelEncoder()
y_labels_encoded = le.fit_transform(y_labels.astype(str))
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_labels_encoded), y=y_labels_encoded)
class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)

model = model.to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.000001, weight_decay=1e-5)

# Training loop
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []
num_epochs = 25
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
        imgs = imgs.to(device)
        labels = labels.to(device).long()
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    train_loss = running_loss / total
    train_acc = correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    # Validation
    model.eval()
    val_running_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
            imgs = imgs.to(device)
            labels = labels.to(device).long()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)
    val_loss = val_running_loss / val_total
    val_acc = val_correct / val_total
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

# Plot learning and accuracy curves
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(range(1, num_epochs+1), train_losses, marker='o', label='Train Loss')
plt.plot(range(1, num_epochs+1), val_losses, marker='x', label='Val Loss')
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.subplot(1,2,2)
plt.plot(range(1, num_epochs+1), train_accuracies, marker='o', label='Train Acc')
plt.plot(range(1, num_epochs+1), val_accuracies, marker='x', label='Val Acc')
plt.title('Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.show()

# Evaluation
model.eval()
correct = 0
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(device)
        labels = labels.to(device).long()
        outputs = model(imgs)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
print(f"Test accuracy: {correct/len(test_ds):.4f}")
