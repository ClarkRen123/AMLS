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
from keras.utils import to_categorical
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def deeptrain():
    photos = load('brain.npy')
    labels = load('class.npy')
    print(photos.shape, labels.shape)
    # create directories
    dataset_home = 'brain_dataset/'
    subdirs = ['train/', 'test/']
    csv_data = pd.read_csv("./dataset/label.csv")
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
    src_directory = './dataset/image/'
    for file in listdir(src_directory):
        src = src_directory + file
        dst_dir = 'train/'
        if random.random() < val_ratio:
            dst_dir = 'test/'
        file_index = img_name.index(file)
        file_label = img_label[file_index]
        dst = dataset_home + dst_dir + file_label + '/' + file
        copyfile(src, dst)
#CNN
def define_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
                     input_shape=(200, 200, 3)))
    model.add(MaxPool2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPool2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPool2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# plot diagnostic learning curves
def summarize_diagnostics(history):
    # plot loss
    plt.subplot(211)
    plt.title('Cross Entropy Loss')
    plt.plot(history.history['loss'], color='blue', label='train')
    plt.plot(history.history['val_loss'], color='orange', label='test')
    # plot accuracy
    plt.subplot(212)
    plt.title('Classification Accuracy')
    plt.plot(history.history['accuracy'], color='blue', label='train')
    plt.plot(history.history['val_accuracy'], color='orange', label='test')
    # save plot to file
    filename = sys.argv[0].split('/')[-1]
    plt.savefig(filename + '_plot.png')
    plt.close()

# run the test harness for evaluating a model
def run_test_harness():
    # define model
    model = define_model()
    # create data generator
    datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    # prepare iterators
    train_it = datagen.flow_from_directory('brain_dataset/train/',
                                            batch_size=64, target_size=(200, 200))
    test_it = datagen.flow_from_directory('brain_dataset/test/',
                                           batch_size=64, target_size=(200, 200))
    # fit model
    history = model.fit(train_it, steps_per_epoch=len(train_it),
                                  validation_data=test_it, validation_steps=len(test_it), epochs=20, verbose=2)
    # evaluate model
    _, acc = model.evaluate(test_it, steps=len(test_it), verbose=0)
    print('accuracy:')
    print('> %.3f' % (acc * 100.0))
    # learning curves
    summarize_diagnostics(history)


deeptrain()
# entry point, run the test harness
run_test_harness()
