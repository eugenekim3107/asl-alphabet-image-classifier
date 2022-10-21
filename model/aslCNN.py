import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np

os.chdir("/Users/eugenekim/PycharmProjects/aslAlphabetClassification")

from data.customDataset import ASLDataset

# CNN Class
class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 30, (5, 5))
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(30, 50, (5, 5))
        self.fc1 = nn.Linear(50 * 4 * 4, 2000)
        self.fc2 = nn.Linear(2000, 1000)
        self.fc3 = nn.Linear(1000, 200)
        self.fc4 = nn.Linear(200, 5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 50 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Load dataset and set batch size
dataset = ASLDataset(csv_file = "data/aslDataset.csv",
                     root_dir = "data/raw_data/asl_alphabet_complete",
                     transform = transforms.ToTensor())

batch_size = 800
train_set, test_set = torch.utils.data.random_split(dataset, [2000, 500])
train_loader = DataLoader(dataset=train_set, batch_size = batch_size, shuffle = True)

# Define Model and Learning Parameters
cnn = CNN()

learning_rate = .01
num_epochs = 50

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(cnn.parameters(), lr=learning_rate)

n_total_steps = len(train_loader)

# Train CNN
for epoch in range(num_epochs):
    total_loss = 0
    for i, (images, labels) in enumerate(train_loader):
        # Forward Propagation
        optimizer.zero_grad()
        outputs = cnn(images)
        labels = labels.type(torch.long)
        loss = criterion(outputs, labels)

        # Backward Propagation
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    if epoch % 10 == 0:
        print(f'Epoch {epoch + 1}, Loss: {total_loss}')
print("Finished Training")

torch.save(cnn.state_dict(), "model/cnn.pth")
