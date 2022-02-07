#IMPORTS
import torch
import torch.nn as nn
import torch.functional as F

import torchvision


def conv2d(inChannels, outChannels, kernel_size=3, padding=1, bias=False):
    return nn.Conv2d(inChannels, outChannels, kernel_size=kernel_size, padding=padding, bias=bias)

def relu():
    return nn.ReLU()

def maxPool2d(kernel_size=2):
    return nn.MaxPool2d(kernel_size=kernel_size)
    
def convTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)


#WHAT HAPPENS IN EVERY STATION
class Block(nn.Module):
    def __init__(self, inChannels, outChannels):
        super().__init__()
        self.conv1 = conv2d(inChannels, outChannels)
        self.conv2 = conv2d(inChannels, outChannels)
        self.relu = relu()
    
    def forward(self, x):
        return self.relu(self.conv2(self.relu(self.conv1(x))))

#DOWNSAMPLING USING THE STATIONS
class Encoder(nn.Module):
    def __init__(self, chs=(3,64,128,256,512,1024)):
        super().__init__()
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)])
        self.pool = maxPool2d()
    
    def forward(self, x):
        ftrs = [] #We need to store the outputs to pass them later into the decoder (U-Net)
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs

#UPSAMPLING
class Decoder(nn.Module):
    def __init__(self, chs=(1024, 512, 256, 128, 64)):
        super().__init__()
        self.chs = chs
        self.upconvs = nn.ModuleList([convTranspose2d(chs[i], chs[i+1]) for i in range(len(chs)-1)])
        self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)]) 
        
    def forward(self, x, encoder_features):
        for i in range(len(self.chs)-1):
            x = self.upconvs[i](x)
            enc_ftrs = self.crop(encoder_features[i], x) #Is necesary?
            x = torch.cat([x, enc_ftrs], dim=1)
            x = self.dec_blocks[i](x)
        return x
    
    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs   = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs

#U-Net
class GeneratorUNetV2(nn.Module):
    def __init__(self, enc_chs=(3,64,128,256,512,1024), dec_chs=(1024, 512, 256, 128, 64), num_class=1):
        super().__init__()
        self.encoder = Encoder(enc_chs)
        self.decoder = Decoder(dec_chs)
        self.head = conv2d(dec_chs[-1], num_class, 1)

    def forward(self, x):
        enc_ftrs = self.encoder(x)
        out = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out = self.head(out)
        return out