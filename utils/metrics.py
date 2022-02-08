from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def get_ssim(image, image_noise):
    return ssim(image, image_noise)
    
def get_psnr(image, image_noise):
    return psnr(image, image_noise)
