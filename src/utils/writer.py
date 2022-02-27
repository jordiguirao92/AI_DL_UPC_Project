import os
import datetime
import torch
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter

class TensorboardLogger():

    def __init__(self, task, model):
        logdir = os.path.join("./logs",f"{task}-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}")
        self.writer = SummaryWriter(log_dir=logdir)

    def log_generator_training(self, model, epoch, loss_train, acc_train, ssim_train, psnr_train, loss_val, acc_val, ssim_val, psnr_val, reconstruction_image):
        self.writer.add_scalar('Generator/train_loss', loss_train, epoch)
        self.writer.add_scalar('Generator/train_acc', acc_train, epoch)
        self.writer.add_scalar('Generator/train_ssim', ssim_train, epoch)
        self.writer.add_scalar('Generator/train_psnr', psnr_train, epoch)

        self.writer.add_scalar('Generator/val_loss', loss_val, epoch)
        self.writer.add_scalar('Generator/val_acc', acc_val, epoch)
        self.writer.add_scalar('Generator/val_ssim', ssim_val, epoch)
        self.writer.add_scalar('Generator/val_psnr', psnr_val, epoch)

        self.writer.add_image('Reconstructed images from the validation set', make_grid(reconstruction_image), epoch)

        for name, weight in model.encoder.named_parameters():
            self.writer.add_histogram(f"{name}/value", weight, epoch)
            self.writer.add_histogram(f"{name}/grad", weight.grad, epoch)

    # TODO Complete for the discriminator training
    def log_discriminator_training(self, epoch, train_loss_avg, val_loss_avg, val_acc_avg, train_acc_avg):
        self.writer.add_scalar('Discriminator/val_loss', val_loss_avg, epoch)
        self.writer.add_scalar('Discriminator/val_acc', val_acc_avg, epoch)
        self.writer.add_scalar('Discriminator/train_loss', train_loss_avg, epoch)
        self.writer.add_scalar('Discriminator/train_acc', train_acc_avg, epoch)

    def log_model_graph(self, model, train_loader):
        batch, _ = next(iter(train_loader))
        self.writer.add_graph(model, batch)

    # TODO Checkou embeddings
    def log_embeddings(self, model, train_loader, device):
        list_latent = []
        list_images = []
        for i in range(10):
            batch, _ = next(iter(train_loader))
            list_latent.append(model.encoder(batch.to(device)))
            list_images.append(batch)
        latent = torch.cat(list_latent)
        images = torch.cat(list_images)
        self.writer.add_embedding(latent, images)


