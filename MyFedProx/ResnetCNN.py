import torch.nn as nn
import torch.nn.functional as F
import torch

"""
This file represents the CNN we will use for the SIIM_ISIC dataset
"""

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5, 2, padding=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 5, 2, padding=2)
        self.maxpool1 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 5, 2, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, 5, 2, padding=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, 5, 2, padding=2)
        # self.maxpool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(4096, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 1)



    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool1(F.relu(self.conv2(x)))
        x = F.relu(self.bn2(self.conv3(x)))
        x = F.relu(self.bn3(self.conv4(x)))
        x = F.relu(self.conv5(x))
        x = x.view(-1, 4096)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        
        x = nn.Sigmoid()(x)

        return x