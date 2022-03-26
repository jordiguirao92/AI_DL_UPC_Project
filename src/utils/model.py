import torch
import numpy as np
from utils.metrics import get_ssim, get_psnr

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_model(model, path):
    checkpoint = {
        "model_state_dict": model.cpu().state_dict(),
    } 
    torch.save(checkpoint, path)


# GENERATOR TRAINING FUNCTIONS
def train_epoch_generator(train_loader, model, optimizer, criterion):
    model.train()
    losses, ssims, psnrs = [], [], []

    for x, y in train_loader:
        optimizer.zero_grad()
        x, y = x.to(get_device()), y.to(get_device())
        y_ = model(x)
        loss = criterion(y_, y)
        loss.backward()
        optimizer.step()
        
        ssim = get_ssim(y, y_)
        psnr = get_psnr(y, y_)

        losses.append(loss.item())
        ssims.append(ssim.item())
        psnrs.append(psnr.item())
    return np.mean(losses), np.mean(ssims), np.mean(psnrs)

def eval_epoch_generator(eval_loader, model, criterion):
    model.eval()
    losses, ssims, psnrs = [], [], []

    for x, y in eval_loader:
        x, y = x.to(get_device()), y.to(get_device())
        with torch.no_grad():
            y_ = model(x)
            loss = criterion(y_, y)
            ssim = get_ssim(y, y_)
            psnr = get_psnr(y, y_)

            losses.append(loss.item())
            ssims.append(ssim.item())
            psnrs.append(psnr.item())
    return np.mean(losses), np.mean(ssims), np.mean(psnrs), y_



# GAN TRAINING FUNCTIONS
def train_epoch_GAN(train_loader, model_g, model_d, optimizer_g, optimizer_d, criterion_g, criterion_d, d_weight):
    model_g.train()
    model_d.train()
    losses_d, losses_g, ssims, psnrs = [], [], [], []

    for noisy_real, clean_real in train_loader:
        noisy_real, clean_real = noisy_real.to(get_device()), clean_real.to(get_device())
        clean_fake = model_g(noisy_real)
        '''
        Discriminator
        '''
        optimizer_d.zero_grad()
        #Train with fake
        fake_ab = torch.cat((noisy_real, clean_fake), 1)
        pred_fake = model_d.forward(fake_ab.detach())
        loss_d_fake = criterion_d(pred_fake, False)

        #Train with real
        real_ab = torch.cat((noisy_real, clean_real), 1)
        pred_real = model_d.forward(real_ab)
        loss_d_real = criterion_d(pred_real, True)

        #Comined D Loss
        loss_d = (loss_d_fake + loss_d_real) *0.5

        loss_d.backward()
        optimizer_d.step()

        '''
        Generator
        '''
        optimizer_g.zero_grad()
        fake_ab = torch.cat((noisy_real, clean_fake), 1)
        pred_fake = model_d.forward(fake_ab)
        loss_g_gan = criterion_d(pred_fake, True)

        loss_g_l1 = criterion_g(clean_fake, clean_real) * d_weight
        
        loss_g = loss_g_gan + loss_g_l1 #Loss normal (L1) + lo que viene del generador (clasificar como true lo que es fake)

        loss_g.backward()
        optimizer_g.step()


        ssim = get_ssim(clean_real, clean_fake)
        psnr = get_psnr(clean_real, clean_fake)

        losses_d.append(loss_d.item())
        losses_g.append(loss_g.item())
        ssims.append(ssim.item())
        psnrs.append(psnr.item())
    return np.mean(losses_d), np.mean(losses_g), np.mean(ssims), np.mean(psnrs)


def eval_epoch_GAN(eval_loader, model_g, criterion):
  model_g.eval()
  eval_losses, ssims, psnrs = [], [], []
  for noisy_real, clean_real in eval_loader:
    noisy_real, clean_real = noisy_real.to(get_device()), clean_real.to(get_device())
    with torch.no_grad():
        output = model_g(noisy_real)

        loss = criterion(output, clean_real)
        ssim = get_ssim(clean_real, output)
        psnr = get_psnr(clean_real, output)

        eval_losses.append(loss.item())
        ssims.append(ssim.item())
        psnrs.append(psnr.item())
  return np.mean(eval_losses), np.mean(ssims), np.mean(psnrs), output