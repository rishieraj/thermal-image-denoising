from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms

class ThermalDataset(Dataset):
    """Thermal image dataset loader"""
    def __init__(self, csv_path, img_size=(480, 640)):
        self.df = pd.read_csv(csv_path)
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor()
        ])

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        raw_img = Image.open(row['raw_path']).convert('L')
        clean_img = Image.open(row['clean_path']).convert('L')
        return self.transform(raw_img), self.transform(clean_img)
    
    def __len__(self):
        return len(self.df)

def get_loaders(config):
    """Create train and test dataloaders"""
    full_dataset = ThermalDataset(config['data_path'])
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    return random_split(full_dataset, [train_size, test_size])