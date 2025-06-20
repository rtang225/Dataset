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
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        lat = self.data.iloc[idx]['latitude']
        lon = self.data.iloc[idx]['longitude']
        target = np.log1p(self.data.iloc[idx]['area'])  # Use regression target
        img = get_image_from_coords(lat, lon)
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(target, dtype=torch.float32)

# Load dataset
csv_path = 'regimagelabels.csv'
dataset = ImageCoordDataset(csv_path)

# Train/test split
indices = np.arange(len(dataset))
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
train_ds = torch.utils.data.Subset(dataset, train_idx)
test_ds = torch.utils.data.Subset(dataset, test_idx)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=64)

# Simple CNN regression model
class SimpleCNNRegressor(nn.Module):
    def __init__(self):
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
        self.fc3 = nn.Linear(128, 1)
    def forward(self, x):
        x = self.pool1(torch.sigmoid(self.conv1(x)))
        x = self.pool2(torch.sigmoid(self.conv2(x)))
        x = self.pool3(torch.sigmoid(self.conv3(x)))
        x = self.pool4(torch.sigmoid(self.conv4(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout1(torch.relu(self.fc1(x)))
        x = self.dropout2(torch.relu(self.fc2(x)))
        x = self.fc3(x)
        return x.squeeze(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNNRegressor().to(device)

# Training setup for regression
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.000001, weight_decay=1e-5)

# Training loop
train_losses = []
val_losses = []
num_epochs = 25
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for imgs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
        imgs = imgs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
    train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(train_loss)

    # Validation
    model.eval()
    val_running_loss = 0.0
    with torch.no_grad():
        for imgs, targets in tqdm(test_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            val_running_loss += loss.item() * imgs.size(0)
    val_loss = val_running_loss / len(test_loader.dataset)
    val_losses.append(val_loss)
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

# Plot learning curves
plt.figure(figsize=(8,5))
plt.plot(range(1, num_epochs+1), train_losses, marker='o', label='Train Loss')
plt.plot(range(1, num_epochs+1), val_losses, marker='x', label='Val Loss')
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.tight_layout()
plt.show()

# Evaluation: print final validation loss
model.eval()
preds = []
targets_all = []
with torch.no_grad():
    for imgs, targets in test_loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        preds.append(outputs.cpu().numpy())
        targets_all.append(targets.cpu().numpy())
preds = np.concatenate(preds)
targets_all = np.concatenate(targets_all)
#preds_inverted = np.expm1(preds)
#targets_inverted = np.expm1(targets_all)
preds_inverted = preds
targets_inverted = targets_all
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(targets_inverted, preds_inverted)
r2 = r2_score(targets_inverted, preds_inverted)
print(f'Final Validation Loss: {val_losses[-1]:.4f}')
print(f'Final MSE (inverted): {mse:.4f}')
print(f'Final R^2 Score (inverted): {r2:.4f}')
