import torch
import matplotlib.pyplot as plt
from utils.metrics import get_history_metrics
from utils.dataLoader import get_data_loaders
from utils.model import get_device
from IPython import embed
from torchvision import transforms
import cv2

def get_plot_loss():
    loss_history_train_g, loss_history_val_g, loss_history_train_d, loss_history_val_d = get_history_metrics('loss')
    plt.title("Training&Validation loss")
    plt.plot(loss_history_train_g, label='train_g')
    plt.plot(loss_history_train_d, label='train_d')
    plt.plot(loss_history_val_g, label='val_g')
    #plt.plot(loss_history_val_d, label='val_d')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def get_plot_image(model):
    '''
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
    '''
    transform_inference = transforms.Compose([transforms.ToPILImage(), transforms.Resize(500), transforms.ToTensor()])
    original_noisy = cv2.imread("C:/Users/34619/DeepLearning/Posgraduate/AI_DL_UPC_Project/dataset/images/image_2_7_rgb_noise.png", 1)
    original_clean = cv2.imread("C:/Users/34619/DeepLearning/Posgraduate/AI_DL_UPC_Project/dataset/images/image_2_7_rgb.png", 1)
    original_noisy = cv2.cvtColor(original_noisy, cv2.COLOR_BGR2RGB)
    original_clean = cv2.cvtColor(original_clean, cv2.COLOR_BGR2RGB)
    #Prepare Data
    t_original_noisy = transform_inference(original_noisy)
    t_original_noisy = torch.unsqueeze(t_original_noisy, dim=0)

    #Output Generator + Discriminator
    output_gd = model(t_original_noisy.to(get_device()))
    output_gd = torch.squeeze(output_gd)
    output_gd = output_gd.cpu().detach().numpy().transpose(1,2,0)
    #output_gd = cv2.cvtColor(output_gd, cv2.COLOR_BGR2RGB)

    images = [original_clean, original_noisy,output_gd]
    fig, axs = plt.subplots(1, 3, figsize=[25, 20])
    for ix in range(3):
        axs[ix].imshow(images[ix])
    plt.show()