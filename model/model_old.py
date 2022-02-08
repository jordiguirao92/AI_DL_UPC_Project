import torch
import torch.nn as nn
import torch.nn.functional as F


def conv2d(inChannels, outChannels, kernel_size=3, padding=1, bias=False):
    return nn.Conv2d(inChannels, outChannels, kernel_size=kernel_size, padding=padding, bias=bias)

def relu():
    return nn.ReLU()

def maxPool2d(kernel_size=2):
    return nn.MaxPool2d(kernel_size=kernel_size)
    
def convTranspose2d(in_channels, out_channels, kernel_size=2, stride=1, padding=0):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

class BlockDown(nn.Module):
    def __init__(self, inChannels, outChannels):
        super().__init__
        self.inChannels = inChannels
        self.outChannels = outChannels

        self.conv1 = conv2d(self.inChannels, self.outChannels)
        self.conv2 = conv2d(self.inChannels, self.outChannels)
        self.relu = relu()
        self.pool = maxPool2d()

    def forward(self, x):
        return self.pool(self.relu(self.conv2(self.conv1(x))))


class BlockUp(nn.Module):
    def __init__(self, inChannels, outChannels):
        super().__init__
        self.inChannels = inChannels
        self.outChannels = outChannels

        self.conv1 = conv2d(self.inChannels, self.outChannels)
        self.conv2 = conv2d(self.inChannels, self.outChannels)
        self.relu = relu()
        self.pool = convTranspose2d(self.inChannels, self.outChannels)

    def forward(self, x):
        return self.pool(self.relu(self.conv2(self.conv1(x))))


class Encoder(nn.Module):
    def __init__(self):
        super().__init__
        self.block1 = BlockDown(3, 32)
        self.block2 = BlockDown(32, 64)
        self.block3 = BlockDown(64, 128)
        self.block4 = BlockDown(128, 256)
        self.block5 = BlockDown(256, 512)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super().__init__
        self.block1 = BlockUp(512, 1024)
        self.block2 = BlockUp(1024, 512)
        self.block3 = BlockUp(512, 256)
        self.block4 = BlockUp(256, 128)
        self.block5 = BlockUp(128, 64)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x

class Output(nn.Module):
    def __init__(self, inChannels, outChannels):
        super().__init__
        self.inChannels = inChannels
        self.outChannels = outChannels

        self.conv1 = conv2d(64, 32)
        self.conv2 = conv2d(32, 2, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class GeneratorUnet(nn.Module):
    def __init__(self):
        super().__init__
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.output = Output()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.output(x)
        return x