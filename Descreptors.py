from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
import cv2 as cv

def hog_descriptor(img):
    resized_img1 = resize(img, (128*4, 64*4))
    fd1, hog_image1 = hog(resized_img1, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), visualize=True, multichannel=True)
    return fd1

def hue_moment_descriptor(img):
    _,im = cv.threshold(im, 128, 255, cv.THRESH_BINARY)
    # Calculate Moments 
    moments = cv.moments(im) 
# Calculate Hu Moments 
    huMoments = cv.HuMoments(moments)
    for i in range(0,7):
        huMoments[i] = -1* copysign(1.0, huMoments[i]) * log10(abs(huMoments[i]))
    return huMoments    