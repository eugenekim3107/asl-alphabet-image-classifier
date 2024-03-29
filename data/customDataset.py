import os
import pandas as pd
import scipy.ndimage
import torch
from torch.utils.data import Dataset
import cv2
from data.filter import detect_edge

class ASLDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.chartonum = {"A":0, "B":1, "C":2, "D":3, "del":4, "E":5, "F":6,
                "G":7,"H":8,"I":9,"J":10,"K":11,"L":12,"M":13,"N":14,
                "nothing":15,"O":16,"P":17,"Q":18,"R":19,"S":20,
                 "space":21,"T":22,"U":23,"V":24,"W":25,"X":26,"Y":27,"Z":28}
        self.numtochar = dict([(value, key) for key, value in self.chartonum.items()])

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        letter_path = self.annotations.iloc[index, 1]
        file_dir = os.path.join(self.root_dir, self.numtochar[letter_path])
        img_path = os.path.join(file_dir, self.annotations.iloc[index, 0])
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (28,28))
        image = scipy.ndimage.gaussian_filter(image, 1)
        y_label = torch.tensor(self.annotations.iloc[index, 1], dtype=torch.uint8)
        if self.transform:
            image = self.transform(image)
        return image, y_label
