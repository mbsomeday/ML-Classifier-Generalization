import numpy as np
import os, torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image



class noise_dataset(Dataset):
    def __init__(self, num):
        self.num = num
        self.images = torch.randn(self.num, 3, 224, 224)
        print(f'Generated {self.num} noise samples.')

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        data_dict = {
            'image': self.images[idx]
        }
        return data_dict



























