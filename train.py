import time
import pandas as pd
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import os
import cv2
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn import preprocessing
from skimage.transform import resize
from skimage.io import imread
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
import pickle
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings('always')  # "error", "ignore", "always", "default", "module" or "once"


def train():

    categories = ["meningioma_tumor", "glioma_tumor", "pituitary_tumor", "no_tumor"]
    data = pd.read_csv("./dataset/label.csv")
    img_label = data['label'].tolist()
    img_name = data['file_name'].tolist()
    image_dir = "./dataset/image"
    flat_data_arr = []
    target_arr = []
    samplenum = int(input('input int number from 100 to 3000 for how many samples to be used in training.\n '
                    'Large sample number yields high accuracy model, but it might takes days to train it.\n'
                          'if you do not have enough time, smaller sample size is recommended'))
    print(f'loading category')
    picnum = 0
    for img in os.listdir(image_dir):
        img_array = imread(os.path.join(image_dir, img))
        img_resized = resize(img_array, (50, 50, cv2.INTER_AREA))
        flat_data_arr.append(img_resized.flatten())
        target_index = img_name.index(img)
        target_arr.append(categories.index(img_label[target_index]))
        picnum += 1
        if picnum == samplenum:
            break
    flat_data = np.array(flat_data_arr)
    target = np.array(target_arr)
    df = pd.DataFrame(flat_data)
    df['Target'] = target
    print(df)
    print(f'loaded category successfully')

    x = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=0,stratify=y)
    print('Splitted Successfully')
    print(y)
    print(x)
    choice = int(input('please choose classifier: 1. KNN ; 2. SVM ; 3. MLP '))
    if choice == 2:
        param_grid={'C':[5, 10, 100],'gamma':[0.1, 1, 10],'kernel':['linear','rbf','poly']}
        svc = svm.SVC(probability=True)
        print("Training begin")
        model = GridSearchCV(svc,param_grid, n_jobs=-1)
        start = time.time()
    elif choice == 1:
        print("Training begin")
        model = KNeighborsClassifier(n_neighbors=3)
        start = time.time()
    elif choice == 3:
        '''param_grid = {
            'hidden_layer_sizes': [(50,), (100,), (150,)],
            'activation': ['logistic','tanh', 'relu'],
            'solver': ['sgd', 'adam','lbfgs'],
        }
        mlp = MLPClassifier(random_state=1, max_iter=500)
        print('Training begin')
        model = GridSearchCV(mlp, param_grid, n_jobs=-1)'''
        model = MLPClassifier(random_state=1, max_iter=500)
        start = time.time()
    model.fit(x_train, y_train)
    end = time.time()
    if choice == 2:
        print('Training completed with best parameter:')
        print(model.best_params_)
    y_pred = model.predict(x_test)
    print("The predicted Data is :")
    print(y_pred)
    print("The actual data is:")
    print(np.array(y_test))
    classification_report(y_pred,y_test)
    print(f"The model is {accuracy_score(y_pred,y_test)*100}% accurate")
    print('time elapsed: ', (end-start))
    #confusion_matrix(y_pred,y_test)
    #pickle.dump(model,open('img_model.p','wb'))
    #print("Pickle is dumped successfully")


train()
