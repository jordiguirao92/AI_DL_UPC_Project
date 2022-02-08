import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time
from torchvision import transforms
from torch.utils.data import DataLoader
from .utils.model import get_device, save_model, binary_accuracy
from .model.model import GeneratorUNet
from .utils.plot import get_plot_loss
from .utils.metrics import get_ssim, get_psnr


loss_history_train = []
loss_history_val = []

def train_model(config):
    # TODO Define train/test loaders
    #train_set = datasets.Dataset(root="", train=True, download=True, transform=transform)
    #test_set = datasets.Dataset(root="", train=False, download=True, transform=transform)
    #train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True)
    #test_loader = DataLoader(test_set, batch_size=config["batch_size"], shuffle=True)

    model = GeneratorUNet().to(get_device)

    # Optimizer: optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    
    # Criterion: BCEWithLogitsLoss(), nn.CrossEntropyLoss(), nn.MSELoss(), F.mse_loss(denoised, noisy_target, reduction='sum')
    criterion = nn.MSELoss()
    
    start_time = time.time()

    for epoch in range(config["epochs"]):
        loss, acc, ssim, psnr = train_epoch(train_loader, model, optimizer, criterion)
        loss_history_train.append(loss)
        print(f"Train Epoch {epoch} loss={loss:.2f} acc={acc:.2f}, ssim={ssim:.2f}, psnr={psnr:.2f}")
        
        loss, acc, ssim, psnr = eval_epoch(my_model, val_loader, criterion)
        loss_history_val.append(loss)
        print(f"Eval Epoch {epoch} loss={loss:.2f} acc={acc:.2f}, ssim={ssim:.2f}, psnr={psnr:.2f}")
    
    end_time = time.time()
    train_time = end_time - start_time
    print(f"The training take {train_time / 60} minutes")

    print("Generate plot")
    get_plot_loss(loss_history_train, loss_history_val)
    return model
    pass


# UTILS Functions
def train_epoch(train_loader, model, optimizer, criterion):
    model.train()
    accs, losses = [], []
    ssims, psnrs = [], []

    for step, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()
        x, y = x.to(get_device), y.to(get_device)
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


def eval_epoch(val_loader, model, criterion):
    accs, losses = [], []
    ssims, psnrs = [], []

    with torch.no_grad():
        model.eval()
        for x, y in val_loader:
            x, y = x.to(get_device), y.to(get_device)
            y_ = model(x)
            y = y.unsqueeze(1).float()
            loss = criterion(y_, y)

            acc = binary_accuracy(y, y_)
            ssim = get_ssim(y, y_)
            psnr = get_psnr(y, y_)

            losses.append(loss.item())
            accs.append(acc.item())
            ssims.append(ssim.item())
            psnrs.append(psnr.item())
    return np.mean(losses), np.mean(accs), np.mean(ssims), np.mean(psnrs)


if __name__ == "__main__":
    config = {
        "lr": 1e-3,
        "batch_size": 100,
        "epochs": 5,
    }
    generator = train_model(config)
    #save_model(generator, 'generator.pt')
