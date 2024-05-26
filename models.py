import os

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.models import resnet50
from torchvision.models.resnet import ResNet50_Weights
from PIL import Image


class SimpleResNet(nn.Module):
    def __init__(self, num_classes, load_path=None):
        super().__init__()
        self.base_model = resnet50(weights=None)
        if load_path is not None:
            self.base_model.load_state_dict(torch.load(load_path))
        else:
            self.base_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        num_ftrs = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.base_model(x)


class ResNetFeatureExtractor(nn.Module):
    def __init__(self, load_path=None):
        super().__init__()
        self.base_model = resnet50(weights=None)
        if load_path is not None:
            self.base_model.load_state_dict(torch.load(load_path))
        else:
            self.base_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        self.base_model.fc = nn.Identity()

    def forward(self, x):
        features = self.base_model(x)
        return features


class ImageFolderToTensor(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, img_path


def extract_features(image_folder_path, batch_size=32, image_size=(256, 256), load_path=None):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = ImageFolderToTensor(image_folder_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = ResNetFeatureExtractor()
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    features = []
    paths = []
    with torch.no_grad():
        for images, image_paths in dataloader:
            images = images.to(device)
            outputs = model(images)
            features.append(outputs.cpu().detach())
            paths.extend(image_paths)

    features = torch.cat(features, dim=0)
    return features, paths
