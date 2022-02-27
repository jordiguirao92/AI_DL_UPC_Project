import torch
import matplotlib.pyplot as plt
from utils.metrics import get_history_metrics
from utils.dataLoader import get_data_loaders
from utils.model import get_device
from IPython import embed

def get_plot_loss():
    loss_history_train, loss_history_val = get_history_metrics('loss')
    plt.title("Training&Validation loss")
    plt.plot(loss_history_train, label='train')
    plt.plot(loss_history_val, label='val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def get_plot_image(model):
    train_loader, test_loader = get_data_loaders(batch_size=1)
    images, labels = next(iter(train_loader))
    plt.title("Image input")
    plt.imshow(images[0].cpu().detach().numpy().transpose(1,2,0))
    plt.show()

    t_img = torch.unsqueeze(images[0], dim=0)
    prediction = model(t_img.to(get_device()))

    plt_image = torch.squeeze(prediction)
    image_prepare = plt_image.cpu().detach().numpy().transpose(1,2,0)
    plt.title("Image output")
    plt.imshow(image_prepare)
    plt.show()

