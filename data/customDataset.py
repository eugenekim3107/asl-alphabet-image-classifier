import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import cv2

class ASLDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.numtochar = {"A":0, "B":1, "C":2, "D":3, "del":4, "E":5, "F":6,
                "G":7,"H":8,"I":9,"J":10,"K":11,"L":12,"M":13,"N":14,
                "nothing":15,"O":16,"P":17,"Q":18,"R":19,"S":20,
                 "space":21,"T":22,"U":23,"V":24,"W":25,"X":26,"Y":27,"Z":28}
        self.chartonum = dict(map(reversed, self.numtochar.items()))

    def __len(self):
        return len(self.annotations)

    def __getitem__(self, index):
        file_dir = os.path.join(self.root_dir, self.chartonum[int(self.annotations.iloc[index, 1])])
        img_path = os.path.join(file_dir, self.annotations.iloc[index, 0])
        image = cv2.imread(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index, 11]))

        if self.transform:
            image = self.transform(image)

        return image, y_label