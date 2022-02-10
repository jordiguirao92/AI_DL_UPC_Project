from image_slicer import slice

def slice_images(path, parts):
    return slice(path, parts)

#The path root is from the project root folder
slice_images('./dataset/images/image_01_rgb.png', 12)
slice_images('./dataset/images/image_01_rgb_noise.png', 12)