import pandas as pd
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
from customDataset import ASLDataset

data = ASLDataset("aslDataset.csv", "raw_data/asl_alphabet_complete")
train_loader = DataLoader(data, batch_size = 100)
for batch in train_loader:
    inp, out = batch
    break