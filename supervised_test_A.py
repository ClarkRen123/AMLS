import pandas as pd
import time
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import os
import cv2
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.io import imread
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
import pickle

# import data
data = pd.read_csv("./test/label.csv")
img_label = data['label'].tolist()
img_name = data['file_name'].tolist()
image_dir = "./test/image"
flat_data_arr = []
target_arr = []
count = 0
total = 200

# load model
model = pickle.load(open('supervisedTask_A.p','rb'))
print('test begin, please wait')
start = time.time()

# test the 200 given images
for i in range(200):
  img=imread("./test/image/"+img_name[i])
  img_resize=resize(img,(100,100,cv2.INTER_AREA))
  l=[img_resize.flatten()]
  prediction = model.predict(l)[0]
  if prediction == 0:
    if img_label == "no-tumor":
      count += 1
  else:
    if img_label != "no_tumor":
      count += 1
end = time.time()
Acc = count/total
print("Accuracy of model:", Acc*100,"%")
print("time elapsed:", end-start,'seconds')
