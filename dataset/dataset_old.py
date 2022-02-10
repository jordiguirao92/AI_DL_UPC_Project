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
from image_slicer import slice

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

