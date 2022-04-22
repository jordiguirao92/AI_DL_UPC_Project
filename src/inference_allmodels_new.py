from numpy import diff
import torch
from model.generator import GeneratorUNet
import cv2
from torchvision import transforms
from utils.model import get_device
from matplotlib import pyplot as plt
import torch.nn as nn

big = True
difficulty = "medium"

if big:
    transform_inference = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
else:
    transform_inference = transforms.Compose([transforms.ToPILImage(), transforms.CenterCrop(256), transforms.ToTensor()])

#MODELS
model1_path = "./checkpoints/gan-20220420-200407-1/model_g-20220420-204211.pt"
model2_path = "./checkpoints/generator-20220420-203616-2.pt"
model3_path = "./checkpoints/gan-20220421-04226-3/model_g-20220421-050520.pt"
model4_path = "./checkpoints/gan-20220420-212329-4/model_g-20220420-212329.pt"
model5_path = "./checkpoints/gan-20220421-073439-5/model_g-20220421-073439.pt"
model6_path = "./checkpoints/gan-20220421-063956-6/model_g-20220421-063956.pt"
model7_path = "./checkpoints/gan-20220421-082534-7/model_g-20220421-082534.pt"
model8_path = "./checkpoints/model_g-20220421-153743.pt"
model9_path = "./checkpoints/model_g-20220421-163014.pt"
model10_path = "./checkpoints/model_g-20220421-213633.pt"
model11_path = "./checkpoints/model_g-20220421-203349.pt"
model12_path = "./checkpoints/model_g-20220421-231856.pt"
model13_path = "./checkpoints/model_g-20220421-231856.pt"
model14_path = "./checkpoints/model_g-20220422-173925.pt"
model15_path = "./checkpoints/model_g-20220422-173146.pt"


#LOAD MODELS
checkpoint1 = torch.load(model1_path)
model_g1 = GeneratorUNet(normalization=nn.Sigmoid(), normalization_layer="batch")
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

checkpoint8 = torch.load(model8_path)
model_g8 = GeneratorUNet(normalization=nn.Tanh(), normalization_layer="spectral")
model_g8.load_state_dict(checkpoint8["model_state_dict"])

checkpoint9 = torch.load(model9_path)
model_g9 = GeneratorUNet(normalization=nn.Tanh(), normalization_layer="spectral")
model_g9.load_state_dict(checkpoint9["model_state_dict"])

checkpoint10 = torch.load(model10_path)
model_g10 = GeneratorUNet(normalization=nn.Tanh(), normalization_layer="spectral")
model_g10.load_state_dict(checkpoint10["model_state_dict"])

checkpoint11 = torch.load(model11_path)
model_g11 = GeneratorUNet(normalization=nn.Tanh(), normalization_layer="spectral")
model_g11.load_state_dict(checkpoint11["model_state_dict"])

checkpoint12 = torch.load(model12_path)
model_g12 = GeneratorUNet(normalization=nn.Tanh(), normalization_layer="spectral")
model_g12.load_state_dict(checkpoint12["model_state_dict"])

checkpoint13 = torch.load(model13_path)
model_g13 = GeneratorUNet(normalization=nn.Tanh(), normalization_layer="spectral")
model_g13.load_state_dict(checkpoint13["model_state_dict"])

checkpoint14 = torch.load(model14_path)
model_g14 = GeneratorUNet(normalization=nn.Tanh(), normalization_layer="spectral")
model_g14.load_state_dict(checkpoint14["model_state_dict"])

checkpoint15 = torch.load(model15_path)
model_g15 = GeneratorUNet(normalization=nn.Tanh(), normalization_layer="spectral")
model_g15.load_state_dict(checkpoint15["model_state_dict"])

#INPUT DATA
if difficulty=="easy":
    noisy_path = f"./dataset/images/image_302_5_rgb_noise.png"
    clean_path = f"./dataset/images/image_302_5_rgb.png"
if difficulty=="medium":
    noisy_path = f"./dataset/images/image_312_9_rgb_noise.png"
    clean_path = f"./dataset/images/image_312_9_rgb.png"
if difficulty=="hard":
    noisy_path = f"./dataset/images/image_313_9_rgb_noise.png"
    clean_path = f"./dataset/images/image_313_9_rgb.png"

original_noisy = cv2.imread(noisy_path, 1)
original_clean = cv2.imread(clean_path, 1)
original_noisy = cv2.cvtColor(original_noisy, cv2.COLOR_BGR2RGB) #Aplicar centercrop
original_clean = cv2.cvtColor(original_clean, cv2.COLOR_BGR2RGB) #Aplicar centercrop
#APPLY TRANSFORM AND ADAPT DIMENSIONS TO FIT INTO THE MODEL
t_original_noisy_plot = transform_inference(original_noisy)
t_original_clean_plot = transform_inference(original_clean)
t_original_noisy = torch.unsqueeze(t_original_noisy_plot, dim=0)

#OUTPUTS OF MODELS
if big:
    models = [model_g2, model_g10]
    device = "cpu"
else:
    models = [model_g1, model_g2, model_g3, model_g4, model_g5, model_g6, model_g7, model_g8, model_g9, model_g10, model_g11, model_g12, model_g13]
    device = get_device()

outputs = []

#CREATE OUTPUT OF ALL MODELS
for model in models:
    model.to(device)
    output = model(t_original_noisy.to(device))
    output = torch.squeeze(output)
    output = output.cpu().detach().numpy().transpose(1,2,0)
    outputs.append(output)
    del model
    torch.cuda.empty_cache()

#SHOW OUTPUTS OF ALL MODELS
counter = 1
for output in outputs:
    plt.imshow(output)
    plt.title(f"Model {counter}")
    plt.show()
    #plt.imsave(f"./checkpoints/outputs/model{counter}_{big}_{difficulty}.png", output)
    counter+=1

#PLOT IN A SINGLE IMAGE ALL OUTPUTS
fig, axs = plt.subplots(3, 4, figsize=[25,20])
row = 0
col = 0
for ix in range(len(outputs)):
    axs[row, col].imshow(outputs[ix])
    axs[row, col].set_title(f"Model {ix+1}")
    col +=1
    if (ix+1)%4 == 0:
        row += 1
        col = 0
plt.show()

#COMPARE ONE OUTPUT WITH INPUTS
ix_output = 1
compare_outputs = [t_original_noisy_plot.cpu().detach().numpy().transpose(1,2,0), t_original_clean_plot.cpu().detach().numpy().transpose(1,2,0), outputs[ix_output-1], outputs[1]]
fig, axs = plt.subplots(1, 4, figsize=[25,20])
for ix in range(len(compare_outputs)):
    axs[ix].imshow(compare_outputs[ix])
    if ix==0:
        axs[ix].set_title("Noisy")
    elif ix==1:
        axs[ix].set_title("Clean")
    else:
        axs[ix].set_title(f"Model {ix_output}")
plt.show()