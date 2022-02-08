import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time
from torchvision import transforms
from torch.utils.data import DataLoader
from .utils.model import get_device, save_model, binary_accuracy
from .model.model import GeneratorUnet
from .model.modelV2 import GeneratorUNetV2
from .utils.plot import get_plot_loss


loss_history_train = []
loss_history_val = []

def train_model(config):
    # TODO Define train/test loaders
    #...
    model = GeneratorUnet().to(get_device)
    # TODO Define optimizer
    # Adam(unet.parameters(), lr=config.INIT_LR)
    # optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    #optimizer = 
    
    # TODO Define the criterion
    # lossFunc = BCEWithLogitsLoss()
    # criterion = nn.CrossEntropyLoss()
    #criterion = 
    
    start_time = time.time()

    for epoch in range(config["epochs"]):
        loss, acc = train_epoch(train_loader, model, optimizer, criterion, hparams)
        loss_history_train.append(loss)
        print(f"Train Epoch {epoch} loss={loss:.2f} acc={acc:.2f}")
        
        loss, acc = eval_epoch(my_model, val_loader)
        loss_history_val.append(loss)
        print(f"Eval Epoch {epoch} loss={loss:.2f} acc={acc:.2f}")
    
    end_time = time.time()
    train_time = end_time - start_time
    print(f"The training take {train_time / 60} minutes")

    print("Generate plot")
    get_plot_loss(loss_history_train, loss_history_val)
    return model
    pass


# UTILS Functions
def train_epoch(train_loader, model, optimizer, criterion, hparams):
    model.train()
    accs, losses = [], []

    for step, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()
        x, y = x.to(get_device), y.to(get_device)
        y_ = model(x)
        loss = criterion(y_, y)
        loss.backward()
        optimizer.step()
        acc = binary_accuracy(y, y_)
        losses.append(loss.item())
        accs.append(acc.item())
    return np.mean(losses), np.mean(accs)


def eval_epoch(val_loader, model):
    accs, losses = [], []
    with torch.no_grad():
        model.eval()
        for x, y in val_loader:
            x, y = x.to(get_device), y.to(get_device)
            y_ = model(x)
            y = y.unsqueeze(1).float()
            loss = F.binary_cross_entropy_with_logits(y_, y)
            acc = binary_accuracy(y, y_)
            losses.append(loss.item())
            accs.append(acc.item())
    return np.mean(losses), np.mean(accs)


if __name__ == "__main__":
    config = {
        "lr": 1e-3,
        "batch_size": 100,
        "epochs": 5,
    }
    generator = train_model(config)
    #save_model(generator, 'generator.pt')
