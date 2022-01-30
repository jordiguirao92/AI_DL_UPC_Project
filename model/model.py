import torch
import torch.nn as nn
import torch.nn.functional as F


# TODO Define the model

class MyModel(nn.Module):

    def __init__(self, h1=32, h2=64, h3=128, h4=256, h5=256):
        super().__init__()
        self.conv1 = nn.Conv2d(3, h1, 3, padding=1)
        self.conv2 = nn.Conv2d(h1, h2, 3, padding=1)
        self.conv3 = nn.Conv2d(h2, h3, 3, padding=1)
        self.conv4 = nn.Conv2d(h3, h4, 3, padding=1)

        self.fc1 = nn.Linear(8*8*h4, h5)
        self.fc2 = nn.Linear(h5, 1)

        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = torch.flatten(x, start_dim=1)

        x = F.relu(self.fc1(x))
        return self.fc2(x)
