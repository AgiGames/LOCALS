import os
import ast
import numpy as np
import torch
from torch.utils.data import Dataset
import pickle
import cv2

image_size = 448

with open(r"original_dataset.pkl", "rb") as f:
    data = pickle.load(f)
    
def fix_label(label: np.ndarray): 
    fixed_label = np.zeros(label.shape)
    num_rows = label.shape[0]
    num_cols = label.shape[1]
    
    for i in range(num_rows):
        for j in range(num_cols):
            xn, yn, c = label[i][j]
            fixed_label[i][j][0] = ((xn - (j / num_cols)) / (1 / num_cols)) * c
            fixed_label[i][j][1] = ((yn - (i / num_rows)) / (1 / num_rows)) * c
            fixed_label[i][j][2] = c
            
    return fixed_label

images_list = []
images_dir = 'converted-dataset/images'
numpy_dump_dir = 'converted-dataset/labels'
file_count = 0
for broken_image, broken_label in data:
    converted_image = (broken_image * 255).astype(np.uint8)
    converted_image = cv2.cvtColor(converted_image, cv2.COLOR_BGR2RGB)
    fixed_label = fix_label(broken_label)
    cv2.imwrite(f'converted-dataset/images/{file_count}.png', converted_image)
    images_list.append(f'{file_count}.png')
    np.save(f'converted-dataset/labels/{file_count}.npy', fixed_label)
    
    file_count += 1
    
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
    
dataset = Pencils(images_list, images_dir, numpy_dump_dir)
torch.save(dataset, 'locals-for-pencils-dataset.pt')