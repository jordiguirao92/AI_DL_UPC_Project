import torch
import random
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import argparse

from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision import datasets
from .config import config
from .utils.noises import AddGaussianNoise, AddGaussianNoiseSkImage, AddSaltNoise, AddSpeckleNoise, AddSaltPeperNoise, AddPoissonNoise

print(config)
# Define Transform
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), AddGaussianNoise()]) # transforms.Normalize((0.5, ), (0.5,))

# Define train and test dataset
train_dataset = datasets.ImageNet('../data/train',train=True, download=True, transform=transform)
test_dataset = datasets.ImageNet('../data/val',train=False, download=True, transform=transform)

# Define train and test loaders
train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=True)

# Get the datasets for train and test
def get_datasets():
    return train_dataset, test_dataset

# Get the dataloader for train and test
def get_dataloader():
    return train_loader, test_loader

# Visualize a random image from a dataset
def visualize_dataset_image(dataset):
    random_image = random.randint(0, len(dataset))
    #x, _ = dataset[7777]
    #plt.imshow(x.numpy()[0], cmap='gray')
    plt.imshow(dataset[random_image])
    plt.title(f"Example #{random_image}")
    plt.axis('off')
    plt.show()

# Visualize a batch from the dataloader
def visualize_batch(dataloader, classes, dataset_type):
	# get batch
    batch = next(iter(dataloader))
	# initialize a figure
	fig = plt.figure(f"{dataset_type} batch")
	figsize=(config["batch_size"], config["batch_size"])
	# loop over the batch size
	for i in range(0, config["batch_size"]):
		# create a subplot
		ax = plt.subplot(2, 4, i + 1)
		# grab the image, convert it from channels first ordering to
		# channels last ordering, and scale the raw pixel intensities
		# to the range [0, 255]
		image = batch[0][i].cpu().numpy()
		image = image.transpose((1, 2, 0))
		image = (image * 255.0).astype("uint8")
		# grab the label id and get the label from the classes list
		idx = batch[1][i]
		label = classes[idx]
		# show the image along with the label
		plt.imshow(image[..., 0], cmap="gray")
		plt.title(label)
		plt.axis("off")
	# show the plot
	plt.tight_layout()
	plt.show()


print("[INFO] Visualize a dataset image...")
visualize_dataset_image(train_dataset)

print("[INFO] Visualize a batch...")
visualize_batch(train_loader, train_dataset.classes, "train")



##Proposal Eva

import cv2
import os
import numpy as np
from torch.utils.data import Dataset
"""
Esto va un poco a gustos; puedes tener una única carpeta con las imágenes siguiendo una nomenclatura que te permita asociar fácilmente las parejas. Por ejemplo "imagen_123_rgb.png" e "imagen_123_rgb_ruido.png".
Luego puedes construirte una lista .txt listando todas las imágenes "*_rgb.png". De hecho, sería recomendable que te genraras 3 listas: training.txt, validation.txt y testing.txt.

"""
class NoiseDataset(Dataset):

    def __init__(self, path_to_images, mode='training'):
 
        file_to_partition_list = os.path.join(path_to_images, f"{mode}.txt")
        self.filenames = np.readtxt(file_to_partition_list, type='str')
        self.mode = mode
        
        if mode == "training":
            np.random.shuffle(self.filenames)
        

    def __len__(self):
        return self.filenames.shape[0]

    def __getitem__(self, idx):

        filename_rgb = self.filenames[idx]
        filename_rgb_noise = filename_rgb.replace('_rgb.png', '_rgb_ruido.png')
        nimg = cv2.imread(filename_rgb)
        nimg_noise = cv2.imread(filename_rgb_noise)

        return nimg, nimg_noise