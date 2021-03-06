import numpy as np
from ignite.metrics import SSIM, PSNR
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

loss_history_train_g = []
loss_history_val_g = []
loss_history_train_d = []
loss_history_val_d = []

ssim_history_train = []
ssim_history_val = []

psnr_history_train = []
psnr_history_val = []


def get_ssim(image, image_noise):
    ssim = SSIM(data_range=1.0)
    ssim.reset()
    ssim.update((image_noise, image))
    ssim_index = ssim.compute()
    return ssim_index
    
def get_psnr(image, image_noise):
    psnr = PSNR(data_range=1.0)
    psnr.reset()
    psnr.update((image_noise, image))
    psnr_index = psnr.compute()
    return psnr_index


def update_history_metrics_g(mode, loss, ssim, psnr):
    if mode == 'validation':
        loss_history_val_g.append(loss)
        ssim_history_val.append(ssim)
        psnr_history_val.append(psnr)
    elif mode == 'training':
        loss_history_train_g.append(loss)
        ssim_history_train.append(ssim)
        psnr_history_train.append(psnr)
    else:
        raise Exception('Please indicate a correct mode in argument: validation or training')
    
    return loss, ssim, psnr

def update_history_metrics_d(mode, loss):
    if mode == 'validation':
        loss_history_val_d.append(loss)
    elif mode == 'training':
        loss_history_train_d.append(loss)
    else:
        raise Exception('Please indicate a correct mode in argument: validation or training')
    
    return loss

def get_history_metrics(metric):
    if metric == 'loss':
        return loss_history_train_g, loss_history_val_g, loss_history_train_d, loss_history_val_d
    elif metric == 'ssim':
        return ssim_history_train, ssim_history_val
    elif metric == 'psnr':
        return psnr_history_train, psnr_history_val
