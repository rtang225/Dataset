# no change in results
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

# AlexNet model definition
class AlexNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        # Dynamically determine the flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            x = self.features(dummy)
            flat_size = x.view(1, -1).size(1)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(flat_size, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Dataset and transforms

def get_image_from_coords(lat, lon):
    IMAGE_FOLDER = 'images/wildfire'
    filename = f"{lon},{lat}.jpg"
    filepath = os.path.join(IMAGE_FOLDER, filename)
    if os.path.exists(filepath):
        try:
            return Image.open(filepath).convert('RGB')
        except Exception:
            return Image.fromarray(np.uint8(np.random.rand(224, 224, 3) * 255))
    else:
        return Image.fromarray(np.uint8(np.random.rand(224, 224, 3) * 255))

default_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # AlexNet input size
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
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=32)

# Model, loss, optimizer
num_classes = len(pd.read_csv(csv_path)['Area_Class'].unique())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AlexNet(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
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
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

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
