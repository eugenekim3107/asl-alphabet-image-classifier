import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

def noiseFilter(image):
    image_copy = image.copy()

    return image_copy