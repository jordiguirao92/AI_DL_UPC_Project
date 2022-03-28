import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, normalization = "sigmoid", activation = "leakyRelu", normalization_layer = "batch", input_nc = 6):
        super().__init__()
        self.conv1 = nn.Conv2d(input_nc, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=4, padding=1)
        self.convOut = nn.Conv2d(512, 1, kernel_size=4, padding=1)

        self.normalization_layer = normalization_layer
        self.activation = activation
        self.normalization = normalization

        if self.normalization_layer == "batch":
          self.norm1 = nn.BatchNorm2d(128)
          self.norm2 = nn.BatchNorm2d(256)
          self.norm3 = nn.BatchNorm2d(512)
          self.norm4 = nn.BatchNorm2d(512)

        elif self.normalization_layer == "instance":
          self.norm1 = nn.InstanceNorm2d(128)
          self.norm2 = nn.InstanceNorm2d(256)
          self.norm3 = nn.InstanceNorm2d(512)
          self.norm4 = nn.InstanceNorm2d(512)

        elif self.normalization_layer == "spectral":
          self.norm1 = nn.utils.spectral_norm(self.conv2)
          self.norm2 = nn.utils.spectral_norm(self.conv3)
          self.norm3 = nn.utils.spectral_norm(self.conv4)
          self.norm4 = nn.utils.spectral_norm(self.conv5)

        if self.activation == "leakyRelu":
          self.act = nn.LeakyReLU(0.2)
        elif self.activation == "relu":
          self.act = nn.relu()
        
        if self.normalization == "sigmoid":
          self.actout = nn.Sigmoid()
        elif self.normalization == "tanh":
          self.actout = nn.Tanh()
    
    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.norm1(self.conv2(x)))
        x = self.act(self.norm2(self.conv3(x)))
        x = self.act(self.norm3(self.conv4(x)))
        x = self.act(self.norm4(self.conv5(x)))
        x = self.actout(self.convOut(x))
        return x