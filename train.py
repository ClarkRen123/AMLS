import pandas as pd
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
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

'''mod = SourceModule("""
    _global_ void fitter(x_train, y_train)
    {
        model.fit(x_train, y_train)
    }
""")

modelfit = mod.get_function("fitter")'''
#def fitter(model, x, y):
#    return model.fit(x, y)


def train():

    categories = ["meningioma_tumor", "glioma_tumor", "pituitary_tumor", "no_tumor"]
    data = pd.read_csv("C:/Users/clark/Desktop/y4/AMLS/AMLS_I_assignment_kit/dataset/label.csv")
    img_label = data['label'].tolist()
    img_name = data['file_name'].tolist()
    image_dir = "C:/Users/clark/Desktop/y4/AMLS/AMLS_I_assignment_kit/dataset/image"
    flat_data_arr = []
    target_arr = []

    print(f'loading category')
    picnum = 0
    for img in os.listdir(image_dir):
        img_array = imread(os.path.join(image_dir, img))
        img_resized = resize(img_array, (50, 50, cv2.INTER_AREA))
        flat_data_arr.append(img_resized.flatten())
        target_index = img_name.index(img)
        target_arr.append(categories.index(img_label[target_index]))
        picnum += 1
        if picnum == 500:
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
    param_grid={'C':[0.1,1,10,100],'gamma':[0.0001,0.001,0.1,1],'kernel':['rbf','poly']}
    svc = svm.SVC(probability=True)
    print("Training begin")
    #model = GridSearchCV(svc,param_grid, n_jobs=-1)
    model = KNeighborsClassifier(n_neighbors=3)
    print('gridsearchcv done')
    model.fit(x_train, y_train)
    print('Training completed with best parameter:')
    #print(model.best_params_)
    y_pred=model.predict(x_test)
    print("The predicted Data is :")
    print(y_pred)
    print("The actual data is:")
    print(np.array(y_test))
    #classification_report(y_pred,y_test)
    print(f"The model is {accuracy_score(y_pred,y_test)*100}% accurate")
    #confusion_matrix(y_pred,y_test)
    #pickle.dump(model,open('img_model.p','wb'))
    #print("Pickle is dumped successfully")


train()
