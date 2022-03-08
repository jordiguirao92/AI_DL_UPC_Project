import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import time
import datetime
from IPython import embed
from utils.model import get_device, save_model, train_epoch, eval_epoch
from model.generator import GeneratorUNet
from utils.writer import TensorboardLogger
from utils.plot import get_plot_loss, get_plot_image
from utils.dataLoader import get_data_loaders
from utils.metrics import update_history_metrics_g
from utils.parser import args


def train_model(model, config):
    model = model.to(get_device())
    
    #Get train/test loaders
    train_loader, eval_loader = get_data_loaders(config["batch_size"])
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    
    # Criterion
    criterion = config["loss"]

    logger = TensorboardLogger("generator-training", model)
    logger.log_model_graph(model, train_loader)
    
    start_time = time.time()
    print(f"TRAINING GENERATOR START - {datetime.datetime.now()} - Your are training your model using {get_device()}")

    for epoch in range(config["epochs"]):
        loss_train, ssim_train, psnr_train = train_epoch(train_loader, model, optimizer, criterion)
        update_history_metrics_g('training', loss_train, ssim_train, psnr_train)
        print(f"Train Epoch {epoch} loss={loss_train:.2f}, ssim={ssim_train:.2f}, psnr={psnr_train:.2f}")
        
        loss_val, ssim_val, psnr_val, reconstruction_image = eval_epoch(eval_loader, model, criterion)
        update_history_metrics_g('validation', loss_val, ssim_val, psnr_val)
        print(f"Eval Epoch {epoch} loss={loss_val:.2f}, ssim={ssim_val:.2f}, psnr={psnr_val:.2f}")

        logger.log_generator_training(model, epoch, loss_train, ssim_train, psnr_train, loss_val, ssim_val, psnr_val, reconstruction_image)
    
    end_time = time.time()
    train_time = end_time - start_time
    print(f"TRAINING FINISH - {datetime.datetime.now()} - The training take {(train_time / 60):.2f} minutes")

    print(f"GENERATE PLOT LOSS - {datetime.datetime.now()}")
    get_plot_loss()
    
    return model


def generator_init(config):
    print(f"CONFIGURATION PARAMETERS: {config}")
    model = GeneratorUNet().to(get_device())
    generator = train_model(model, config)
    get_plot_image(generator)
    #save_model(generator, f"generator-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.pt")
