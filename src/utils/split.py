from image_slicer import slice
from glob import glob
import shutil
import os

def slice_images(path, parts):
    return slice(path, parts)

#The path root is from the project root folder
#slice_images('dataset\original\image_1_rgb.png', 12)
#slice_images('dataset\original\image_1_rgb_noise.png', 12)

source = "dataset\original"
destination = "dataset\images"
allfiles = glob(source + "/*.PNG")

for file in allfiles:
    slice_images(file, 12)
    
allfiles = glob(source + "/*.PNG")
for file in allfiles:
    if file.endswith("_rgb.PNG") == False and file.endswith("_rgb_noise.PNG") == False:
        shutil.move(file, os.path.join(destination) + os.path.join("/") + os.path.basename(file))
