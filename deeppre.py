import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from matplotlib.image import imread
import pandas as pd
import os
import numpy as np
import cv2
from os import listdir
from numpy import asarray
from numpy import save
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
from skimage.color import rgb2gray
from skimage.transform import resize
from numpy import load

n_c = 4
categories = ["meningioma_tumor", "glioma_tumor", "pituitary_tumor", "no_tumor"]
data = pd.read_csv("C:/Users/clark/Desktop/y4/AMLS/AMLS_I_assignment_kit/dataset/label.csv")
img_label = data['label'].tolist()
img_name = data['file_name'].tolist()
image_dir = "C:/Users/clark/Desktop/y4/AMLS/AMLS_I_assignment_kit/dataset/image"
data_arr = []
target_arr = []
samplenum = int(input('input int number from 100 to 3000 for how many samples to be used in training.\n '
                'Large sample number yields high accuracy model, but it might takes days to train it.\n'
                      'if you do not have enough time, smaller sample size is recommended'))
print(f'loading category')
picnum = 0
# define location of dataset
photos, labels = list(), list()
# enumerate files in the directory
for file in listdir(image_dir):
    # load image
    photo = load_img(os.path.join(image_dir, file), target_size=(200, 200))
    # convert to numpy array
    photo = img_to_array(photo)
    # store
    photos.append(photo)
labels=img_label
# convert to a numpy arrays
photos = asarray(photos)
labels = asarray(labels)
print(photos.shape, labels.shape)
# save the reshaped photos
save('brain.npy', photos)
save('class.npy', labels)