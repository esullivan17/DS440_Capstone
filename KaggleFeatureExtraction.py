import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np
from tqdm import tqdm  # Progress bar

# for loading images from the folder structure
class BrainTumorDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for img_name in os.listdir(class_dir):
                if img_name.endswith(('.jpg', '.png', '.jpeg')):
                    self.samples.append((os.path.join(class_dir, img_name),
                                        self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('L')  # Convert to grayscale
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None, None

        if self.transform:
            image = self.transform(image)

        return image, label

# Custom collate function to skip None samples
def collate_fn(batch):
    batch = [sample for sample in batch if sample[0] is not None]
    if len(batch) == 0:
        return None, None
    return tuple(zip(*batch))

# Feature Extraction CNN Model
class BrainTumorFeatureExtractor(nn.Module):
    def __init__(self, feature_dim=512):
        super(BrainTumorFeatureExtractor, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, feature_dim)

    def forward(self, x, return_all_features=False):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)

        pooled = self.global_pool(x5)
        flattened = pooled.view(pooled.size(0), -1)
        features = self.fc(flattened)

        if return_all_features:
            return {
                'low_level': x1,
                'mid_level1': x2,
                'mid_level2': x3,
                'high_level1': x4,
                'high_level2': x5,
                'global_features': features
            }

        return features

# Function to extract features from images with tqdm
def extract_features(model, dataloader, device):
    model.eval()
    features = []
    labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            if batch[0] is None:
                continue
            inputs, targets = batch
            inputs = torch.stack(inputs).to(device)
            targets = torch.tensor(targets)

            outputs = model(inputs)
            features.extend(outputs.cpu().numpy())
            labels.extend(targets.numpy())

    return np.array(features), np.array(labels)

# Main execution
def main():
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train_dataset = BrainTumorDataset(root_dir='/Users/karthikkuppala/Desktop/archive/Training', transform=transform)
    test_dataset = BrainTumorDataset(root_dir='/Users/karthikkuppala/Desktop/archive/Testing', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4, collate_fn=collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BrainTumorFeatureExtractor(feature_dim=256).to(device)

    train_features, train_labels = extract_features(model, train_loader, device)
    test_features, test_labels = extract_features(model, test_loader, device)

    np.save('train_features.npy', train_features)
    np.save('train_labels.npy', train_labels)
    np.save('test_features.npy', test_features)
    np.save('test_labels.npy', test_labels)

    print(f"Extracted features shapes: Train {train_features.shape}, Test {test_features.shape}")

if __name__ == '__main__':
    main()
