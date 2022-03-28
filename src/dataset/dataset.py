import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
"""
Esto va un poco a gustos; puedes tener una única carpeta con las imágenes siguiendo una nomenclatura que te permita asociar fácilmente las parejas. Por ejemplo "imagen_123_rgb.png" e "imagen_123_rgb_ruido.png".
Luego puedes construirte una lista .txt listando todas las imágenes "*_rgb.png". De hecho, sería recomendable que te genraras 3 listas: training.txt, validation.txt y testing.txt.

"""

# TODO Create the transforms in dataset
class NoiseDataset(Dataset):

    def __init__(self, path_to_images, mode, transform = None):
        file_to_partition_list = os.path.join(path_to_images, f"{mode}.txt")
        self.filenames = np.loadtxt(file_to_partition_list, dtype='str')
        self.mode = mode
        self.path_to_images = path_to_images
        self.transform = transform

    def __len__(self):
        return self.filenames.shape[0]

    def __getitem__(self, idx):
        filename_rgb = self.filenames[idx]
        filename_rgb_noise = filename_rgb.replace('_rgb.png', '_rgb_noise.png')
        nimg = cv2.imread(f"{self.path_to_images}/{filename_rgb}", 1)
        nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
        nimg_noise = cv2.imread(f"{self.path_to_images}/{filename_rgb_noise}", 1)
        nimg_noise = cv2.cvtColor(nimg_noise, cv2.COLOR_BGR2RGB)
        if self.transform:
            nimg = self.transform(nimg)
            nimg_noise = self.transform(nimg_noise)
        return nimg_noise, nimg