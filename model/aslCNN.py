import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
import torch.nn.functional as F

class CNN(nn.Module):

    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, 6, (5,5))
        self.pool = nn.MaxPool2d(5,5)
        self.conv2 = nn.Conv2d(6, 12, (5,5))
        self.fc1 = nn.Linear(12 * 5 * 5, 200)
        self.fc2 = nn.Linear(200, 120)
        self.fc3 = nn.Linear(120, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x