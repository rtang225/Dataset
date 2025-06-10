# diverging training and validation again
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

# LeNet model definition
class LeNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, padding=2)
        self.pool1 = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.AvgPool2d(2, 2)
        # Dynamically determine the flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 56, 56)
            x = torch.relu(self.conv1(dummy))
            x = self.pool1(x)
            x = torch.relu(self.conv2(x))
            x = self.pool2(x)
            flat_size = x.view(1, -1).size(1)
        self.fc1 = nn.Linear(flat_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Dataset and transforms (reuse your pipeline)
def get_image_from_coords(lat, lon):
    IMAGE_FOLDER = 'images/wildfire'
    filename = f"{lon},{lat}.jpg"
    filepath = os.path.join(IMAGE_FOLDER, filename)
    if os.path.exists(filepath):
        try:
            return Image.open(filepath).convert('RGB')
        except Exception:
            return Image.fromarray(np.uint8(np.random.rand(64, 64, 3) * 255))
    else:
        return Image.fromarray(np.uint8(np.random.rand(64, 64, 3) * 255))

default_transform = transforms.Compose([
    transforms.Resize((56, 56)),  # LeNet input size
    transforms.ToTensor(),
])

class ImageCoordDataset(torch.utils.data.Dataset):
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
indices = np.arange(len(dataset))
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
train_ds = Subset(dataset, train_idx)
test_ds = Subset(dataset, test_idx)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=64)

# Model, loss, optimizer
num_classes = len(pd.read_csv(csv_path)['Area_Class'].unique())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LeNet(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

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
        imgs, labels = imgs.to(device), labels.to(device).long()
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
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device).long()
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

# Plot learning curves
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()
plt.subplot(1,2,2)
plt.plot(train_accuracies, label='Train Acc')
plt.plot(val_accuracies, label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Curve')
plt.legend()
plt.tight_layout()
plt.show()

# Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device).long()
        outputs = model(imgs)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
print(f"Test accuracy: {correct/total:.4f}")
