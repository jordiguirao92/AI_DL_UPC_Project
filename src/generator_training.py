import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import time
import datetime
from IPython import embed
from utils.model import get_device, save_model, train_epoch_generator, eval_epoch_generator, test_model_generator
from model.generator import GeneratorUNet
from utils.writer import TensorboardLogger
from utils.plot import get_plot_loss, get_plot_image
from utils.dataLoader import get_data_loaders
from utils.metrics import update_history_metrics_g
from utils.parser import args


def train_model_generator(model, config):
    model = model.to(get_device())
    
    #Get train/test loaders
    train_loader, eval_loader, test_loader = get_data_loaders(config["batch_size"])
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    
    # Criterion
    criterion = config["loss"]

    logger = TensorboardLogger("generator-training", model)
    logger.log_model_graph(model, train_loader)
    
    start_time = time.time()
    print(f"TRAINING GENERATOR START - {datetime.datetime.now()} - Your are training your model using {get_device()}")

    for epoch in range(config["epochs"]):
        loss_train, ssim_train, psnr_train = train_epoch_generator(train_loader, model, optimizer, criterion)
        update_history_metrics_g('training', loss_train, ssim_train, psnr_train)
        if epoch%config['log_interval']==0:
            print(f"Train epoch: {epoch} -- Loss Generator: {loss_train:.2f} -- SSIM: {ssim_train:.2f} -- PSNR: {psnr_train:.2f}")
        
        loss_val, ssim_val, psnr_val, reconstruction_image = eval_epoch_generator(eval_loader, model, criterion)
        update_history_metrics_g('validation', loss_val, ssim_val, psnr_val)
        if epoch%config['log_interval']==0:
            print(f"Eval epoch: {epoch} -- Loss Generator: {loss_val:.2f} -- SSIM: {ssim_val:.2f} -- PSNR: {psnr_val:.2f}")

        logger.log_generator_training(model, epoch, loss_train, ssim_train, psnr_train, loss_val, ssim_val, psnr_val, reconstruction_image)
    
    test_model_generator(test_loader, model, criterion)
    end_time = time.time()
    train_time = end_time - start_time
    print(f"TRAINING FINISH - {datetime.datetime.now()} - The training take {(train_time / 60):.2f} minutes")

    #print(f"GENERATE PLOT LOSS - {datetime.datetime.now()}")
    #get_plot_loss()
    
    return model


def generator_init(config):
    print(f"CONFIGURATION PARAMETERS: {config}")
    model = GeneratorUNet(normalization=config["generator_last"], normalization_layer=config["generator_norm"])
    generator = train_model_generator(model, config)
    #get_plot_image(generator)
    save_model(generator, f"./checkpoints/generator-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.pt")
