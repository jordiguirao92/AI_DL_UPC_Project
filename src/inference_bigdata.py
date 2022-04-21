import torch
from model.generator import GeneratorUNet
import cv2
from torchvision import transforms
from utils.model import get_device
from matplotlib import pyplot as plt
import torch.nn as nn


def inference(image_number = 316):
    #TRANSFORMS TO APPLY
    transform_inference = transforms.Compose([transforms.ToPILImage(), transforms.CenterCrop(256), transforms.ToTensor()])
    transform_image = transforms.Compose([transforms.ToPILImage(), transforms.CenterCrop(256)])

    #PATH TO THE MODEL --> Load the model --> Create new model --> Load weights from the model
    model1_path = "./checkpoints/gan-20220420-200407-1/model_g-20220420-204211.pt"
    model2_path = "./checkpoints/generator-20220420-203616-2.pt"
    model3_path = "./checkpoints/gan-20220421-04226-3/model_g-20220421-050520.pt"
    model4_path = "./checkpoints/gan-20220420-212329-4/model_g-20220420-212329.pt"
    model5_path = "./checkpoints/gan-20220421-073439-5/model_g-20220421-073439.pt"
    model6_path = "./checkpoints/gan-20220421-063956-6/model_g-20220421-063956.pt"
    model7_path = "./checkpoints/gan-20220421-082534-7/model_g-20220421-082534.pt"


    checkpoint1 = torch.load(model1_path)
    model_g1 = GeneratorUNet()
    model_g1.load_state_dict(checkpoint1["model_state_dict"])
    
    checkpoint2 = torch.load(model2_path)
    model_g2 = GeneratorUNet(normalization=nn.Tanh(), normalization_layer="spectral")
    model_g2.load_state_dict(checkpoint2["model_state_dict"])
    
    checkpoint3 = torch.load(model3_path)
    model_g3 = GeneratorUNet(normalization=nn.Tanh(), normalization_layer="spectral")
    model_g3.load_state_dict(checkpoint3["model_state_dict"])
    
    checkpoint4 = torch.load(model4_path)
    model_g4 = GeneratorUNet(normalization=nn.Tanh(), normalization_layer="spectral")
    model_g4.load_state_dict(checkpoint4["model_state_dict"])
    
    checkpoint5 = torch.load(model5_path)
    model_g5 = GeneratorUNet(normalization=nn.Tanh(), normalization_layer="spectral")
    model_g5.load_state_dict(checkpoint5["model_state_dict"])
    
    checkpoint6 = torch.load(model6_path)
    model_g6 = GeneratorUNet(normalization=nn.Tanh(), normalization_layer="spectral")
    model_g6.load_state_dict(checkpoint6["model_state_dict"])
    
    checkpoint7 = torch.load(model7_path)
    model_g7 = GeneratorUNet(normalization=nn.Tanh(), normalization_layer="spectral")
    model_g7.load_state_dict(checkpoint7["model_state_dict"])

    #PATH TO A PAIR OF IMAGES TO APPLY INFERENCE f"Eval epoch: {epoch}
    #noisy_path = f"./dataset/original/image_{image_number}_rgb_noise.PNG"
    #clean_path = f"./dataset/original/image_{image_number}_rgb.PNG"
    noisy_path = f"./dataset/images/image_312_9_rgb_noise.png"
    clean_path = f"./dataset/images/image_312_9_rgb.png"

    #READ IMAGES AND CONVERT TO RGB (TO DISPLAY LATER WITH MATPLOTLIB)
    original_noisy = cv2.imread(noisy_path, 1)
    original_clean = cv2.imread(clean_path, 1)
    original_noisy = cv2.cvtColor(original_noisy, cv2.COLOR_BGR2RGB)
    original_clean = cv2.cvtColor(original_clean, cv2.COLOR_BGR2RGB)
    transformed_original_noisy = transform_image(original_noisy)
    transformed_original_clean =transform_image(original_clean)

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
    #MODEL7
    model_g7.to(get_device())
    output_g7 = model_g7(t_original_noisy.to(get_device()))
    output_g7 = torch.squeeze(output_g7)
    output_g7 = output_g7.cpu().detach().numpy().transpose(1,2,0)


    #DISPLAY IMAGES
    plt.imshow(original_clean)
    plt.title("Original clean")
    plt.show()
    plt.imshow(transformed_original_clean)
    plt.title("T_Original clean")
    plt.show()
    plt.imshow(original_noisy)
    plt.title("Original noisy")
    plt.show()
    plt.imshow(transformed_original_noisy)
    plt.title("T_original_noisy")
    plt.show()
    plt.imshow(output_g1)
    plt.title("output_g1")
    plt.show()
    plt.imshow(output_g2)
    plt.title("output_g2")
    plt.show()
    plt.imshow(output_g3)
    plt.title("output_g3")
    plt.show()
    plt.imshow(output_g4)
    plt.title("output_g4")
    plt.show()
    plt.imshow(output_g5)
    plt.title("output_g5")
    plt.show()
    plt.imshow(output_g6)
    plt.title("output_g6")
    plt.show()
    plt.imshow(output_g7)
    plt.title("output_g7")
    plt.show()


    images = [original_clean, original_noisy, output_g4]
    fig, axs = plt.subplots(1, 3, figsize=[25, 20])
    for ix in range(3):
        axs[ix].imshow(images[ix])
    plt.show()

    images_models = [output_g1, output_g2, output_g3, output_g4, output_g5, output_g6, output_g7]
    fig, axs = plt.subplots(2, 4, figsize=[25, 20])
    for ix in range(7):
        if ix < 3:
            axs[0, ix].imshow(images_models[ix])
        else:
            axs[1, ix-3].imshow(images_models[ix])
    plt.show()

inference()