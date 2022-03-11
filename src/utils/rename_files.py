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
    source = "dataset/images/"
    allfiles = glob(source + "/*.png", recursive = True)
    allfiles.sort()

    for file in allfiles:
        if re.search('noise', file):
            splited = file.split("_")
            os.rename(file, os.path.join(source) + os.path.join("/") + "image_" + str(splited[1]) + "_" + str(splited[-1].replace(".png", "")) + "_rgb_noise.PNG")
        else:
            splited = file.split("_")
            os.rename(file, os.path.join(source) + os.path.join("/") + "image_" + str(splited[1]) + "_" + str(splited[-1].replace(".png", "")) + "_rgb.PNG")
