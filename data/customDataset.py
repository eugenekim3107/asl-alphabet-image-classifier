import cv2
import os
import numpy as np

os.chdir("/data")

numerical_map = {"A":0, "B":1, "C":2, "D":3, "del":4, "E":5, "F":6,
                "G":7,"H":8,"I":9,"J":10,"K":11,"L":12,"M":13,"N":14,
                "nothing":15,"O":16,"P":17,"Q":18,"R":19,"S":20,
                 "space":21,"T":22,"U":23,"V":24,"W":25,"X":26,"Y":27,"Z":28}

file_path = "./raw_data"
train_path = "asl_alphabet_train"

def asl_data_csv(num_image):

    # Final array
    final = None

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

            read_image = cv2.imread(
                os.path.join(file_path, train_path, letter, image))
            temp_image = cv2.cvtColor(read_image, cv2.COLOR_BGR2RGB).flatten()
            full = np.append(temp_image, map_letter)
            count += 1

            if final is None:
                final = full

            else:
                final = np.vstack((final, full))

    np.savetxt("aslDataset.csv", final, delimiter=",")
