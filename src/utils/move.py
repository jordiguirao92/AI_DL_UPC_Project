import shutil
import os
from glob import glob

def move_images():
    source = "dataset/original/"
    destination = "dataset/images/"
    #destination = "dataset\images" #windows base
    #We use here .png, because the slce functions create the images with .png
    allfiles = glob(source + "/*.png")
    for file in allfiles:
        if file.endswith("_rgb.png") == False and file.endswith("_rgb_noise.png") == False:
            shutil.move(file, os.path.join(destination) + os.path.join("/") + os.path.basename(file))

