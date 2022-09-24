import pandas as pd
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
from customDataset import ASLDataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Load Dataset
dataset = ASLDataset(csv_file = "aslDataset.csv", root_dir = "raw_data/asl_alphabet_complete")
batch_size = 50
train_set, test_set = torch.utils.data.random_split(dataset, [1200, 250])
train_loader = DataLoader(dataset=train_set, batch_size = batch_size, shuffle = True)
for batch in train_loader:
    inp, out = batch
    inp = np.array(inp)
    out = np.array(out)
    break

folder = "testPictures/"
cv2.imwrite(os.path.join(folder, "test.jpg"), inp[0])

