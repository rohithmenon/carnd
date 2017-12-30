import cv2
import numpy as np

def preprocess_image(img):
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    shape = grayscale.shape
    return np.reshape(grayscale, (shape[0], shape[1], 1))
