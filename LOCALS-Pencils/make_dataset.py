import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

images_dir = 'converted-dataset/images'
numpy_dump_dir = 'converted_dataset/labels'
image_size = 448

class Pencils(Dataset):
    def __init__(self, image_list, images_dir, numpy_dump_dir):

        self.image_list = image_list
        self.images_dir = images_dir
        self.numpy_dump_dir = numpy_dump_dir

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]

        image = cv2.imread(
            os.path.join(self.images_dir,
                         image_name)
        )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (image_size, image_size))
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)

        prefix = image_name.split('.')[0]
        annotation_matrix = np.load(os.path.join(self.numpy_dump_dir, f'{prefix}.npy'))

        return image, annotation_matrix

images_list = os.listdir(images_dir)
dataset = Pencils(images_list, images_dir, numpy_dump_dir)
torch.save(dataset, 'locals-for-pencils-dataset.pt')