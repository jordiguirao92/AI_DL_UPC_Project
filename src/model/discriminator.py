import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, normalization = "sigmoid", activation = "leakyRelu", normalization_layer = "batch", output_size = 14, input_nc = 6):
        super().__init__()
        '''
        self.conv1 = nn.Conv2d(input_nc, 64, kernel_size=4, stride=2, padding=1) #128
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1) #64
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1) #32
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1) #16
        self.conv5 = nn.Conv2d(512, 512, kernel_size=4, padding=1)#15
        self.convOut = nn.Conv2d(512, 1, kernel_size=4, padding=1)#14
        '''
        '''
        self.conv1 = nn.Conv2d(input_nc, 64, kernel_size=4, stride=2, padding=1) #128
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1) #64
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1) #32
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1) #16
        self.convOut = nn.Conv2d(512, 1, kernel_size=4, padding=1)#15
        Sigmoid
        '''
        self.normalization_layer = normalization_layer
        self.activation = activation
        self.normalization = normalization
        self.output_size = output_size

        self.conv1 = nn.Conv2d(input_nc, 64, kernel_size=4, stride=2, padding=1) #128
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1) #64
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1) #32
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1) #16
        if self.output_size == 15:
          self.conv5 = nn.Conv2d(512, 512, kernel_size=4, padding=1) #15
        else:
          self.conv5 = nn.Conv2d(512, 512, kernel_size=4, stride = 2, padding=1)#8
        self.conv6 = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1) #4
        self.conv7 = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1) #2
        self.convOut = nn.Conv2d(512, 1, kernel_size=4, padding=1) #-1

        if self.normalization_layer == "batch":
          self.norm1 = nn.BatchNorm2d(128)
          self.norm2 = nn.BatchNorm2d(256)
          self.norm3 = nn.BatchNorm2d(512)
          self.norm4 = nn.BatchNorm2d(512)
          self.norm5 = nn.BatchNorm2d(512)
          self.norm6 = nn.BatchNorm2d(512)

        elif self.normalization_layer == "instance":
          self.norm1 = nn.InstanceNorm2d(128)
          self.norm2 = nn.InstanceNorm2d(256)
          self.norm3 = nn.InstanceNorm2d(512)
          self.norm4 = nn.InstanceNorm2d(512)
          self.norm5 = nn.InstanceNorm2d(512)
          self.norm6 = nn.InstanceNorm2d(512)

        elif self.normalization_layer == "spectral":
          self.norm1 = nn.utils.spectral_norm(self.conv2)
          self.norm2 = nn.utils.spectral_norm(self.conv3)
          self.norm3 = nn.utils.spectral_norm(self.conv4)
          self.norm4 = nn.utils.spectral_norm(self.conv5)
          self.norm5 = nn.utils.spectral_norm(self.conv6)
          self.norm6 = nn.utils.spectral_norm(self.conv7)

        if self.activation == "leakyRelu":
          self.act = nn.LeakyReLU(0.2)
        elif self.activation == "relu":
          self.act = nn.relu()
        
        if self.normalization == "sigmoid":
          self.actout = nn.Sigmoid()
        elif self.normalization == "tanh":
          self.actout = nn.Tanh()
    
    def forward(self, x):
      if self.output_size == 14:
        x = self.act(self.conv1(x))
        x = self.act(self.norm1(self.conv2(x)))
        x = self.act(self.norm2(self.conv3(x)))
        x = self.act(self.norm3(self.conv4(x)))
        x = self.act(self.norm4(self.conv5(x)))
        x = self.actout(self.convOut(x))
      elif self.output_size == 15:
        x = self.act(self.conv1(x))
        x = self.act(self.norm1(self.conv2(x)))
        x = self.act(self.norm2(self.conv3(x)))
        x = self.act(self.norm3(self.conv4(x)))
        x = self.actout(self.convOut(x))
      elif self.output_size == 1:
        x = self.act(self.conv1(x))
        x = self.act(self.norm1(self.conv2(x)))
        x = self.act(self.norm2(self.conv3(x)))
        x = self.act(self.norm3(self.conv4(x)))
        x = self.act(self.norm4(self.conv5(x)))
        x = self.act(self.norm5(self.conv6(x)))
        x = self.act(self.norm6(self.conv7(x)))
        x = self.actout(self.convOut(x))
      return x