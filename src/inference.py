import torch
from model.generator import GeneratorUNet
import cv2
from torchvision import transforms
from utils.model import get_device
from matplotlib import pyplot as plt


def inference(image_number = 126):
    #TRANSFORMS TO APPLY
    transform_inference = transforms.Compose([transforms.ToPILImage(), transforms.Resize(256), transforms.ToTensor()])

    #PATH TO THE MODEL --> Load the model --> Create new model --> Load weights from the model
    model_path = "./checkpoints/model_g-20220312-121935.pt"
    checkpoint = torch.load(model_path)
    model_g = GeneratorUNet()
    model_g.load_state_dict(checkpoint["model_state_dict"])

    #PATH TO A PAIR OF IMAGES TO APPLY INFERENCE f"Eval epoch: {epoch}
    noisy_path = f"./dataset/original/image_{image_number}_rgb_noise.PNG"
    clean_path = f"./dataset/original/image_{image_number}_rgb.PNG"

    #READ IMAGES AND CONVERT TO RGB (TO DISPLAY LATER WITH MATPLOTLIB)
    original_noisy = cv2.imread(noisy_path, 1)
    original_clean = cv2.imread(clean_path, 1)
    original_noisy = cv2.cvtColor(original_noisy, cv2.COLOR_BGR2RGB)
    original_clean = cv2.cvtColor(original_clean, cv2.COLOR_BGR2RGB)

    #APPLY TRANSFORM AND ADAPT DIMENSIONS TO FIT INTO THE MODEL
    t_original_noisy = transform_inference(original_noisy)
    t_original_noisy = torch.unsqueeze(t_original_noisy, dim=0)

    #PASS THE IMAGE THROUGH THE MODEL AND ADAPT IT AGAIN TO DISPLAY IT WITH MATPLOTLIB
    model_g.to(get_device())
    output_g = model_g(t_original_noisy.to(get_device()))
    output_g = torch.squeeze(output_g)
    output_g = output_g.cpu().detach().numpy().transpose(1,2,0)
    output_g = cv2.cvtColor(output_g, cv2.COLOR_BGR2RGB)

    #DISPLAY IMAGES
    plt.imshow(original_clean)
    plt.show()
    plt.imshow(original_noisy)
    plt.show()
    plt.imshow(output_g)
    plt.show()
    images = [original_clean, original_noisy, output_g]
    fig, axs = plt.subplots(1, 3, figsize=[25, 20])
    for ix in range(3):
        axs[ix].imshow(images[ix])
    plt.show()



inference()