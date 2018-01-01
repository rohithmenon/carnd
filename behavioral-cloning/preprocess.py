import cv2
import numpy as np

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(5,5))

def preprocess_image1(img):
    resized = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    grayscale = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    shape = grayscale.shape
    #edge_detected = cv2.Canny(grayscale, 50, 120)
    return np.reshape(grayscale, (shape[0], shape[1], 1))

def preprocess_image(img):
    return img
    #grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #shape = grayscale.shape
    #return np.reshape(grayscale, (shape[0], shape[1], 1))
