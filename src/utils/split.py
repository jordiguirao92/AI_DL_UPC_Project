from image_slicer import slice
from glob import glob

#The path root is from the project root folder

def slice_images(path, parts):
    return slice(path, parts)

def slice_database():
    source = "dataset/original/"
    #source = "dataset\original" #windows base
    
    allfiles = glob(source + "/*.PNG")
    for file in allfiles:
        slice_images(file, 12)
