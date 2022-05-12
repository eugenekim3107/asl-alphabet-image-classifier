import pandas as pd
import os
import numpy as np

os.chdir("/Users/eugenekim/PycharmProjects/aslAlphabetClassification/data")

numerical_map = {"A":0, "B":1, "C":2, "D":3, "del":4, "E":5, "F":6,
                "G":7,"H":8,"I":9,"J":10,"K":11,"L":12,"M":13,"N":14,
                "nothing":15,"O":16,"P":17,"Q":18,"R":19,"S":20,
                 "space":21,"T":22,"U":23,"V":24,"W":25,"X":26,"Y":27,"Z":28}

file_path = "./raw_data"
train_path = "asl_alphabet_complete"

def asl_data_csv(num_image):

    # Final array
    final = np.array([])

    for letter in sorted(os.listdir(os.path.join(file_path, train_path))):

        count = 0

        if letter == ".DS_Store":
            continue

        map_letter = numerical_map[letter]

        for image in sorted(
                os.listdir(os.path.join(file_path, train_path, letter))):

            if count == num_image:
                break

            if not image.endswith(".jpg"):
                continue

            count += 1

            if len(final) == 0:
                final = np.array([image, map_letter])

            else:
                final = np.vstack((final, np.array([image, map_letter])))

    return final

data = pd.DataFrame(asl_data_csv(50)).set_index(0)
data.to_csv("aslDataset.csv")
