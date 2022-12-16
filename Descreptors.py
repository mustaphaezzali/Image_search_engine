from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
import numpy as np 
import cv2 as cv

def hog_descriptor(img):
    resized_img1 = resize(img, (128*4, 64*4))
    fd1, hog_image1 = hog(resized_img1, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), visualize=True, multichannel=True)
    return fd1

def hue_moment_descriptor(im):
    #preprocessing
    img = cv.cvtColor(im, cv.COLOR_RGB2GRAY)
    img_len,img_width = img.shape[0],img.shape[1]
    centre = img_len//2,img_width//2
    seuil = img[centre[0]-2:centre[0]+2,centre[1]-2:centre[1]+2].mean()                     #img[img_dim[0]//2,img_dim[1]//2]
    couleur = 255
    options = [cv.THRESH_BINARY,cv.THRESH_BINARY_INV,cv.THRESH_TRUNC,cv.THRESH_TOZERO,cv.THRESH_TOZERO_INV]
    result = cv.threshold(img, int(seuil), couleur, options[0])[1]
    # Calculate Moments 
    moments = cv.moments(result) 
    # Calculate Hu Moments 
    huMoments = cv.HuMoments(moments)
    return huMoments.reshape((7,))    
    