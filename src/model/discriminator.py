import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, input_nc = 6):
        super().__init__()
        self.conv1 = nn.Conv2d(input_nc, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=4, padding=1)
        self.convOut = nn.Conv2d(512, 1, kernel_size=4, padding=1)

        self.batchnorm1 = nn.BatchNorm2d(128)
        self.batchnorm2 = nn.BatchNorm2d(256)
        self.batchnorm3 = nn.BatchNorm2d(512)
        self.act = nn.LeakyReLU(0.2)
        self.actout = nn.Sigmoid()
    
    def forward(self, x):
        #SI NO FUNCIONA PROBAR CON F.LEAKYRELU
        x = self.act(self.conv1(x))
        x = self.act(self.batchnorm1(self.conv2(x)))
        x = self.act(self.batchnorm2(self.conv3(x)))
        x = self.act(self.batchnorm3(self.conv4(x)))
        x = self.act(self.batchnorm3(self.conv5(x)))
        x = self.actout(self.convOut(x))
        return x