import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import time
import datetime
from IPython import embed
from utils.model import get_device, save_model, train_epoch_GAN, eval_epoch_GAN
from model.generator import GeneratorUNet
from model.discriminator import Discriminator
from model.ganLoss import GANLoss
from utils.writer import TensorboardLogger
from utils.plot import get_plot_loss, get_plot_image
from utils.dataLoader import get_data_loaders
from utils.metrics import update_history_metrics_g, update_history_metrics_d
from utils.parser import args


def train_GAN(model_g, model_d, config):
  #Generator
  optimizer_g = optim.Adam(model_g.parameters(), lr=config["lr"])
  model_g = model_g.to(get_device())
  #Discriminator 
  optimizer_d = optim.Adam(model_d.parameters(), lr =config["lr"])
  model_d = model_d.to(get_device())

  #Get train/test loaders
  train_loader, eval_loader = get_data_loaders(config["batch_size"])

  #Criterion_d
  criterionGAN = GANLoss().to(get_device()) 
  #Criterion_g
  criterionL1 = config["loss"].to(get_device())
  criterionMSE = nn.MSELoss().to(get_device())

  logger = TensorboardLogger("GAN-training", model_g)
  logger.log_model_graph(model_g, train_loader)

  start = time.time()
  print(f"TRAINING GAN START - {datetime.datetime.now()} - Your are training your model using {get_device()}")

  for epoch in range(config["epochs"]):
    loss_train_d, loss_train_g, ssim_train, psnr_train = train_epoch_GAN(train_loader, model_g, model_d, optimizer_g, optimizer_d, criterion_g=criterionL1, criterion_d=criterionGAN, d_weight=config["d_weight"])
    update_history_metrics_g('training', loss_train_g.item(), ssim_train.item(), psnr_train.item())
    update_history_metrics_d('training', loss_train_d.item())
    if epoch%config['log_interval']==0:
      print(f"Train epoch: {epoch} -- Loss Generator: {loss_train_g:.2f} -- Loss Discriminator: {loss_train_d:.2f} -- SSIM: {ssim_train:.2f} -- PSNR: {psnr_train:.2f}")

    loss_val, ssim_val, psnr_val, reconstruction_image = eval_epoch_GAN(eval_loader, model_g, criterionMSE)
    update_history_metrics_g('validation', loss_val.item(), ssim_val.item(), psnr_val.item())
    if epoch%config['log_interval']==0:
      print(f"Eval epoch: {epoch} -- Loss Generator: {loss_val:.2f} -- SSIM: {ssim_val:.2f} -- PSNR: {psnr_val:.2f}")
    
    logger.log_generator_training(model_g, epoch, loss_train_g, ssim_train, psnr_train, loss_val, ssim_val, psnr_val, reconstruction_image)
    logger.log_discriminator_training(epoch, loss_train_d)

  end = time.time()
  train_time = end-start
  print(f"The training took {(train_time/60):.2f} minutes")

  print(f"GENERATE PLOT LOSS - {datetime.datetime.now()}")
  get_plot_loss()

  return model_g, model_d


def gan_init(config):
    print(f"CONFIGURATION PARAMETERS: {config}")
    model_g = GeneratorUNet(normalization=config["generator_last"], normalization_layer=config["generator_norm"])
    model_d = Discriminator(normalization=config["discriminator_last"], normalization_layer=config["discriminator_norm"], activation=config["discriminator_activation"])
    model_g, model_d = train_GAN(model_g, model_d, config)
    #get_plot_image(model_g)
    #save_model(model_g, f"./checkpoints/model_g-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.pt")
    #save_model(model_d, f"./checkpoints/model_d-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.pt")
    #checkpoint = torch.load("./checkpoints/checkpoint.pt")




