import cv2
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split

import os

class LOCALSDataset(Dataset):
    def __init__(self, images_dir, labels_dir):
        self.image_list = [
            f_name for f_name in os.listdir(images_dir)
            if f_name.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.image_size=448

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]

        image = cv2.imread(
            os.path.join(self.images_dir,
                         image_name)
        )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)

        prefix = image_name.split('.')[0]
        annotation_matrix = np.load(os.path.join(self.labels_dir, f'{prefix}.npy'))

        return image, annotation_matrix
    
    def get_dataloaders(self, train_split = 0.8, test_split = 0.1, batch_size=16):
        assert (
            0 < train_split < 1
            and 0 < test_split < 1
            and train_split + test_split <= 1
        ), "train_split + test_split must not exceed 1."
        
        N = len(self)
        train_size = int(train_split * N)
        test_size = int(N * test_split)
        val_size = N - (train_size + test_size)
        
        train_dataset, test_dataset, val_dataset = random_split(
            self,
            [train_size, test_size, val_size]
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=True,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True
        ) if val_size > 0 else None
        
        if val_size > 0:
            return train_loader, test_loader, val_loader
        return train_loader, test_loader