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
csv_path = 'image_labels.csv'
dataset = ImageCoordDataset(csv_path)

# Train/test split
indices = np.arange(len(dataset))
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
train_ds = torch.utils.data.Subset(dataset, train_idx)
test_ds = torch.utils.data.Subset(dataset, test_idx)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=32)

# Simple CNN model
def get_num_classes(csv_path):
    df = pd.read_csv(csv_path)
    return len(df['Area_Class'].unique())

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, num_classes)
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

num_classes = get_num_classes(csv_path)
model = SimpleCNN(num_classes)

# Training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for imgs, labels in train_loader:
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
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/total:.4f}, Acc: {correct/total:.4f}")

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
