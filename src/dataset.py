import os
import kagglehub
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from skimage.color import rgb2lab
from PIL import Image
import numpy as np

def download_and_prepare_dataset():
    """Download the dataset and return the correct paths for grayscale and color images."""
    print("Downloading dataset...")
    dataset_path = kagglehub.dataset_download("aayush9753/image-colorization-dataset")

    # Ensure we correctly access the `data/` subdirectory
    dataset_path = os.path.join(dataset_path, "data")

    train_gray_folder = os.path.join(dataset_path, "train_black")  # Grayscale images
    train_color_folder = os.path.join(dataset_path, "train_color")  # Corresponding color images

    if not os.path.exists(train_gray_folder) or not os.path.exists(train_color_folder):
        raise FileNotFoundError(f"Dataset folders not found. Expected: {train_gray_folder} and {train_color_folder}")

    print(f"Dataset ready at: {dataset_path}")
    return train_gray_folder, train_color_folder

class OldPhotoDataset(Dataset):
    def __init__(self, gray_dir, color_dir, transform=None):
        self.gray_dir = gray_dir
        self.color_dir = color_dir
        self.gray_paths = sorted([os.path.join(gray_dir, f) for f in os.listdir(gray_dir) if f.endswith(".jpg")])
        self.color_paths = sorted([os.path.join(color_dir, f) for f in os.listdir(color_dir) if f.endswith(".jpg")])
        self.transform = transform

    def __len__(self):
        return len(self.gray_paths)

    def __getitem__(self, idx):
        gray_img = Image.open(self.gray_paths[idx]).convert("L")  # Grayscale input
        color_img = Image.open(self.color_paths[idx]).convert("RGB")  # Color ground truth

        # Convert color image to LAB color space
        color_img = np.array(color_img) / 255.0  # Normalize to [0,1]
        lab = rgb2lab(color_img)  # Convert to LAB color space

        L = np.array(gray_img) / 100.0  # Normalize grayscale image
        L = torch.tensor(L).unsqueeze(0).float()  # Add channel dimension

        AB = lab[:, :, 1:] / 128.0  # Normalize AB channels to [-1,1]
        AB = torch.tensor(AB).permute(2, 0, 1).float()  # Change shape to (2, H, W)

        return L, AB

def get_dataloader(batch_size=8, num_workers=2, shuffle=True):
    """Download dataset and return a DataLoader."""
    train_gray_folder, train_color_folder = download_and_prepare_dataset()
    
    dataset = OldPhotoDataset(gray_dir=train_gray_folder, color_dir=train_color_folder)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    return dataloader

if __name__ == "__main__":
    print("Preparing dataset and DataLoader...")
    dataloader = get_dataloader()
    
    for L, AB in dataloader:
        print(f"Batch loaded - Grayscale shape: {L.shape}, Color shape: {AB.shape}")
        break  # Load one batch to verify
