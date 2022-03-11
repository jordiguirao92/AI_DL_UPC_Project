from image_slicer import slice
from glob import glob

def slice_images(path, parts):
    return slice(path, parts)

#The path root is from the project root folder
#slice_images('dataset\original\image_1_rgb.png', 12)
#slice_images('dataset\original\image_1_rgb_noise.png', 12)

def slice_database():
    source = "dataset/original/"
    #source = "dataset\original" #windows base
    
    allfiles = glob(source + "/*.PNG")
    for file in allfiles:
        slice_images(file, 12)
