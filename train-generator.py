import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime
from torchvision import transforms
from torch.utils.data import DataLoader
from IPython import embed
from utils.model import get_device, save_model, binary_accuracy
from model.model import GeneratorUNet
from utils.plot import get_plot_loss
from utils.metrics import get_ssim, get_psnr
from utils.writer import TensorboardLogger
from dataset.dataset import NoiseDataset

loss_history_train = []
loss_history_val = []

def train_model(model, config):
    model = model.to(get_device())
    #Define train/test loaders
    #transform = transforms.Compose([transforms.Resize(256), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]) 
    transform = transforms.Compose([transforms.CenterCrop(256), transforms.ToTensor()])
    train_set = NoiseDataset(path_to_images='./dataset/images', mode='training', transform=transform)
    test_set = NoiseDataset(path_to_images='./dataset/images', mode='testing', transform=transform)
    train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(test_set, batch_size=config["batch_size"], shuffle=False)

    # Optimizer: optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    
    # Criterion: BCEWithLogitsLoss(), nn.CrossEntropyLoss(), nn.MSELoss(), F.mse_loss(denoised, noisy_target, reduction='sum')
    criterion = nn.MSELoss()

    #logger = TensorboardLogger("generator-training", model)
    #logger.log_model_graph(model, train_loader)
    
    start_time = time.time()
    print(f"TRAINING START - {datetime.datetime.now()} - Your are training your model using {get_device()}")

    for epoch in range(config["epochs"]):
        loss_train, acc_train, ssim_train, psnr_train = train_epoch(train_loader, model, optimizer, criterion)
        loss_history_train.append(loss_train)
        print(f"Train Epoch {epoch} loss={loss_train:.2f} acc={acc_train:.2f}, ssim={ssim_train:.2f}, psnr={psnr_train:.2f}")
        
        loss_val, acc_val, ssim_val, psnr_val = eval_epoch(test_loader, model, criterion)
        loss_history_val.append(loss_val)
        print(f"Eval Epoch {epoch} loss={loss_val:.2f} acc={acc_val:.2f}, ssim={ssim_val:.2f}, psnr={psnr_val:.2f}")

        #logger.log_generator_training(model, epoch, loss_train, acc_train, ssim_train, psnr_train, loss_val, acc_val, ssim_val, psnr_val)
    
    end_time = time.time()
    train_time = end_time - start_time
    print(f"TRAINING FINISH - {datetime.datetime.now()} - The training take {train_time / 60} minutes")

    print("Generate plot")
    get_plot_loss(loss_history_train, loss_history_val)
    
    return model


# UTILS Functions
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
        #ssim = get_ssim(y, y_)
        #psnr = get_psnr(y, y_)

        losses.append(loss.item())
        accs.append(acc.item())
        #ssims.append(ssim.item())
        #psnrs.append(psnr.item())
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
            #ssim = get_ssim(y, y_)
            #psnr = get_psnr(y, y_)

            losses.append(loss.item())
            accs.append(acc.item())
            #ssims.append(ssim.item())
            #psnrs.append(psnr.item())
    return np.mean(losses), np.mean(accs), np.mean(ssims), np.mean(psnrs)


if __name__ == "__main__":
    config = {
        "lr": 1e-3,
        "batch_size": 1,
        "epochs": 3,
    }
    model = GeneratorUNet().to(get_device())
    generator = train_model(model, config)
    #save_model(generator, 'generator.pt')
