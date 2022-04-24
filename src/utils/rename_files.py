from fnmatch import fnmatchcase
from glob import glob
import os
import re


def rename_original_images(): 
    source = "dataset/original/" #linux base
    #source = "dataset\original\" #windows base

    count_gt = 0
    count_noisy = 0

    allfiles = glob(source + "/**/*.PNG", recursive = True)
    allfiles.sort()
    print(len(allfiles))
    for file in allfiles:
        if fnmatchcase(file, "*_GT_*"):
            os.rename(file, os.path.join(source) + os.path.join("/") + f"image_{count_gt}_rgb.PNG")
            count_gt = count_gt + 1
        elif fnmatchcase(file, "*_NOISY_*"):
            os.rename(file, os.path.join(source) + os.path.join("/") + f"image_{count_noisy}_rgb_noise.PNG")
            count_noisy = count_noisy + 1


def rename_sliced_images(): 
    source_path = "dataset/images/"
    images = sorted(glob(source_path+"/*.png"))

    image_number_noisy = 1
    slice_number_noisy = 1
    image_number = 1
    slice_number = 1
    for image in images:
        if fnmatchcase(image, "*_noise_*"):
            os.rename(image, os.path.join(source_path) + os.path.join("/") + f"image_{image_number_noisy}_{slice_number_noisy}_rgb_noise.png")
            slice_number_noisy = slice_number_noisy + 1
            if slice_number_noisy == 13:
                slice_number_noisy = 1
                image_number_noisy = image_number_noisy + 1
        else:
            os.rename(image, os.path.join(source_path) + os.path.join("/") + f"image_{image_number}_{slice_number}_rgb.png")
            slice_number = slice_number + 1
            if slice_number == 13:
                slice_number = 1
                image_number = image_number + 1


