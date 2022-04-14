import torch
from model.generator import GeneratorUNet
import cv2
from torchvision import transforms
from utils.model import get_device
from matplotlib import pyplot as plt
import torch.nn as nn


def inference(image_number = 316):
    #TRANSFORMS TO APPLY
    transform_inference = transforms.Compose([transforms.ToPILImage(), transforms.Resize(256), transforms.ToTensor()])

    #PATH TO THE MODEL --> Load the model --> Create new model --> Load weights from the model
    model1_path = "./checkpoints/model_g-20220407-200005.pt" #FULL OUR DATASET GAN 10 EPOCHS
    model2_path = "./checkpoints/model_g-20220408-205850.pt" #SMALL OUR DATASET GAN 100 EPOCHS
    model3_path = "./checkpoints/model_g-20220410-145239.pt" #FULL GITHUB DATASET GAN 100 EPOCHS
    model4_path = "./checkpoints/model_g-20220411-132800.pt" #SMALL OUR DATASET GAN 50 EPOCHS - Instance
    model5_path = "./checkpoints/generator-20220412-164835.pt" #SMALL OUR DATASET Generator 70 EPOCHS
    model6_path = "./checkpoints/generator-20220413-170528.pt"
    model8_path = "./checkpoints/model_g-20220413-193641.pt"

    checkpoint1 = torch.load(model1_path)
    model_g1 = GeneratorUNet()
    model_g1.load_state_dict(checkpoint1["model_state_dict"])
    checkpoint2 = torch.load(model2_path)
    model_g2 = GeneratorUNet()
    model_g2.load_state_dict(checkpoint2["model_state_dict"])
    checkpoint3 = torch.load(model3_path)
    model_g3 = GeneratorUNet()
    model_g3.load_state_dict(checkpoint3["model_state_dict"])
    checkpoint4 = torch.load(model4_path)
    model_g4 = GeneratorUNet(normalization_layer="instance")
    model_g4.load_state_dict(checkpoint4["model_state_dict"])
    checkpoint5 = torch.load(model5_path)
    model_g5 = GeneratorUNet()
    model_g5.load_state_dict(checkpoint5["model_state_dict"])
    checkpoint6 = torch.load(model6_path)
    model_g6 = GeneratorUNet(normalization=nn.Tanh(), normalization_layer="instance")
    model_g6.load_state_dict(checkpoint6["model_state_dict"])
    checkpoint8 = torch.load(model8_path)
    model_g8 = GeneratorUNet(normalization=nn.Tanh(), normalization_layer="instance")
    model_g8.load_state_dict(checkpoint8["model_state_dict"])

    #PATH TO A PAIR OF IMAGES TO APPLY INFERENCE f"Eval epoch: {epoch}
    #noisy_path = f"./dataset/original/image_{image_number}_rgb_noise.PNG"
    #clean_path = f"./dataset/original/image_{image_number}_rgb.PNG"
    noisy_path = f"C:/Users/34619/DeepLearning/Posgraduate/AI_DL_UPC_Project/dataset/images/image_312_9_rgb_noise.png"
    clean_path = f"C:/Users/34619/DeepLearning/Posgraduate/AI_DL_UPC_Project/dataset/images/image_312_9_rgb.png"

    #READ IMAGES AND CONVERT TO RGB (TO DISPLAY LATER WITH MATPLOTLIB)
    original_noisy = cv2.imread(noisy_path, 1)
    original_clean = cv2.imread(clean_path, 1)
    original_noisy = cv2.cvtColor(original_noisy, cv2.COLOR_BGR2RGB)
    original_clean = cv2.cvtColor(original_clean, cv2.COLOR_BGR2RGB)

    #APPLY TRANSFORM AND ADAPT DIMENSIONS TO FIT INTO THE MODEL
    t_original_noisy = transform_inference(original_noisy)
    t_original_noisy = torch.unsqueeze(t_original_noisy, dim=0)

    #PASS THE IMAGE THROUGH THE MODEL AND ADAPT IT AGAIN TO DISPLAY IT WITH MATPLOTLIB
    #MODEL1
    model_g1.to(get_device())
    output_g1 = model_g1(t_original_noisy.to(get_device()))
    output_g1 = torch.squeeze(output_g1)
    output_g1 = output_g1.cpu().detach().numpy().transpose(1,2,0)
    #output_g = cv2.cvtColor(output_g, cv2.COLOR_BGR2RGB)
    #MODEL2
    model_g2.to(get_device())
    output_g2 = model_g2(t_original_noisy.to(get_device()))
    output_g2 = torch.squeeze(output_g2)
    output_g2 = output_g2.cpu().detach().numpy().transpose(1,2,0)
    #MODEL3
    model_g3.to(get_device())
    output_g3 = model_g3(t_original_noisy.to(get_device()))
    output_g3 = torch.squeeze(output_g3)
    output_g3 = output_g3.cpu().detach().numpy().transpose(1,2,0)
    #MODEL4
    model_g4.to(get_device())
    output_g4 = model_g4(t_original_noisy.to(get_device()))
    output_g4 = torch.squeeze(output_g4)
    output_g4 = output_g4.cpu().detach().numpy().transpose(1,2,0)
    #MODEL5
    model_g5.to(get_device())
    output_g5 = model_g5(t_original_noisy.to(get_device()))
    output_g5 = torch.squeeze(output_g5)
    output_g5 = output_g5.cpu().detach().numpy().transpose(1,2,0)
    #MODEL6
    model_g6.to(get_device())
    output_g6 = model_g6(t_original_noisy.to(get_device()))
    output_g6 = torch.squeeze(output_g6)
    output_g6 = output_g6.cpu().detach().numpy().transpose(1,2,0)
    #MODEL8
    model_g8.to(get_device())
    output_g8 = model_g8(t_original_noisy.to(get_device()))
    output_g8 = torch.squeeze(output_g8)
    output_g8 = output_g8.cpu().detach().numpy().transpose(1,2,0)


    #DISPLAY IMAGES
    plt.imshow(original_clean)
    plt.show()
    plt.imshow(original_noisy)
    plt.show()
    plt.imshow(output_g1)
    plt.show()
    plt.imshow(output_g2)
    plt.show()
    plt.imshow(output_g3)
    plt.show()
    plt.imshow(output_g4)
    plt.show()
    plt.imshow(output_g5)
    plt.show()
    plt.imshow(output_g6)
    plt.show()
    plt.imshow(output_g8)
    plt.show()

    images = [original_clean, original_noisy, output_g4]
    fig, axs = plt.subplots(1, 3, figsize=[25, 20])
    for ix in range(3):
        axs[ix].imshow(images[ix])
    plt.show()

    images_models = [output_g1, output_g2, output_g3, output_g4, output_g5, output_g6, output_g8]
    fig, axs = plt.subplots(2, 4, figsize=[25, 20])
    for ix in range(7):
        if ix < 3:
            axs[0, ix].imshow(images_models[ix])
        else:
            axs[1, ix-3].imshow(images_models[ix])
    plt.show()

inference()