import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

#focusing primarily on these 15 core corruptions npy
# Noise: Gaussian, Shot, Impulse.

# Blur: Defocus, Glass, Motion, Zoom.

# Weather: Snow, Frost, Fog, Brightness.

# Digital: Contrast, Elastic, Pixelate, JPEG Compression.
class CIFAR10C(Dataset):
    def __init__(self, root_dir, corruption_name, severity, transform=None):
        """
        Args:
            root_dir (string): Directory with all the .npy files.
            corruption_name (string): Name of the corruption (e.g., 'fog', 'snow').
            severity (int): Severity level from 1 to 5.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        data_path = os.path.join(root_dir, f"{corruption_name}.npy")
        full_data = np.load(data_path)
        
        label_path = os.path.join(root_dir, "labels.npy")
        full_labels = np.load(label_path)

        start_idx = (severity - 1) * 10_000
        end_idx = severity * 10_000

        self.data = full_data[start_idx: end_idx]
        self.labels = full_labels[start_idx: end_idx]
        self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]

        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long)
    
def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)), #required for ViT-Tiny
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            #mean=[0.485, 0.456, 0.406], #Standard ImageNet mean
            #std=[0.229, 0.224, 0.225] #Standard ImageNet variance
        )
    ])

def get_loader(root_dir, corruption, severity, batch_size=128, input_size=32):
    transform = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.Resize((input_size, input_size)), # Dynamic resize
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    dataset = CIFAR10C(
        root_dir=root_dir,
        corruption_name=corruption,
        severity=severity,
        transform=transform
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)