import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.ndimage

def detect_edge(image, sobel=False):
    image_copy = image.copy()
    if sobel:
        Ix = scipy.ndimage.convolve(image_copy, np.array([[1,0,-1],[2,0,-2],[1,0,-1]]))
        Iy = scipy.ndimage.convolve(image_copy, np.array([[1,2,1],[0,0,0],[-1,-2,-1]]))
    else:
        Ix = scipy.ndimage.convolve(image_copy, np.array([[1,0,-1]]))
        Iy = scipy.ndimage.convolve(image_copy, np.array([[1,0,-1]]).T)
    grad_magnitude = np.sqrt((Ix**2)+(Iy**2))
    return Ix, Iy, grad_magnitude
