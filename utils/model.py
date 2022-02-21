import torch
import numpy as np
from utils.metrics import get_ssim, get_psnr

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_model(model, path):
    torch.save(model.state_dict(), path)

def accuracy(labels, outputs):
    preds = outputs.argmax(-1)
    acc = (preds == labels.view_as(preds)).float().detach().numpy().mean()
    return acc
    
def binary_accuracy(labels, outputs):
    preds = outputs.round()
    acc = (preds == labels.view_as(preds)).float().detach().numpy().mean()
    return acc

def binary_accuracy_with_logits(labels, outputs):
    preds = torch.sigmoid(outputs).round()
    acc = (preds == labels.view_as(preds)).float().detach().numpy().mean()
    return acc

def train_epoch(train_loader, model, optimizer, criterion):
    model.train()
    accs, losses = [], []
    ssims, psnrs = [], []

    for x, y in train_loader:
        optimizer.zero_grad()
        x, y = x.to(get_device()), y.to(get_device())
        y_ = model(x)
        loss = criterion(y_, y)
        loss.backward()
        optimizer.step()
        
        acc = binary_accuracy(y, y_)
        ssim = get_ssim(y, y_)
        psnr = get_psnr(y, y_)

        losses.append(loss.item())
        accs.append(acc.item())
        ssims.append(ssim.item())
        psnrs.append(psnr.item())
    return np.mean(losses), np.mean(accs), np.mean(ssims), np.mean(psnrs)

def eval_epoch(test_loader, model, criterion):
    accs, losses = [], []
    ssims, psnrs = [], []

    with torch.no_grad():
        model.eval()
        for x, y in test_loader:
            x, y = x.to(get_device()), y.to(get_device())
            y_ = model(x)
            loss = criterion(y_, y)
            acc = binary_accuracy(y, y_)
            ssim = get_ssim(y, y_)
            psnr = get_psnr(y, y_)

            losses.append(loss.item())
            accs.append(acc.item())
            ssims.append(ssim.item())
            psnrs.append(psnr.item())
    return np.mean(losses), np.mean(accs), np.mean(ssims), np.mean(psnrs)