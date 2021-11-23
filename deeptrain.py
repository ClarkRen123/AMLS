import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from matplotlib.image import imread
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
import time
import pandas as pd
import os,sys
import numpy as np
import cv2
from os import listdir
import random
from numpy import asarray
from numpy import save
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from skimage.transform import resize
#from skimage.io import imread
from sklearn.model_selection import train_test_split
from skimage.color import rgb2gray
from skimage.transform import resize
from numpy import load
from shutil import copyfile


def deeptrain():
    photos = load('brain.npy')
    labels = load('class.npy')
    print(photos.shape, labels.shape)
    # create directories
    dataset_home = 'AMLSproject/'
    subdirs = ['train/', 'test/']
    csv_data = pd.read_csv("AMLSproject/dataset/label.csv")
    img_label = csv_data['label'].tolist()
    img_name = csv_data['file_name'].tolist()
    for subdir in subdirs:
        # create label subdirectories
        label_dir = ["meningioma_tumor/", "glioma_tumor/", "pituitary_tumor/", "no_tumor/"]
        for labldir in label_dir:
            newdir = dataset_home + subdir + labldir
            os.makedirs(newdir, exist_ok=True)
    # seed random number generator
    random.seed(1)
    # define ratio of pictures to use for validation
    val_ratio = 0.2
    # copy training dataset images into subdirectories
    src_directory = 'AMLSproject/dataset/image/'
    for file in listdir(src_directory):
        src = src_directory + file
        dst_dir = 'train/'
        if random.random() < val_ratio:
            dst_dir = 'test/'
        file_index = img_name.index(file)
        file_label = img_label[file_index]
        dst = dataset_home + dst_dir + file_label + '/' + file
        copyfile(src, dst)





deeptrain()
