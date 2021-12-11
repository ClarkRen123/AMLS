import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.image as image
import cv2
import time
import os
import pandas as pd
import numpy as np


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
torch.cuda.empty_cache()
# model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        # Convolution layer 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        self.batch1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        self.batch2 = nn.BatchNorm2d(32)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1_drop = nn.Dropout(0.25)

        # Convolution layer 2
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.relu3 = nn.ReLU()
        self.batch3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.relu4 = nn.ReLU()
        self.batch4 = nn.BatchNorm2d(64)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2_drop = nn.Dropout(0.25)

        # Convolution layer 3
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0)
        self.relu5 = nn.ReLU()
        self.batch5 = nn.BatchNorm2d(128)

        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=0)
        self.relu6 = nn.ReLU()
        self.batch6 = nn.BatchNorm2d(128)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3_drop = nn.Dropout(0.25)

        # Fully-Connected layer 1
        self.fc1 = nn.Linear(18432, 128)
        self.fc1_relu = nn.ReLU()
        self.dp1 = nn.Dropout(0.5)

        # Fully-Connected layer 2
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # conv layer 1
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.batch1(out)

        out = self.conv2(out)
        out = self.relu2(out)
        out = self.batch2(out)

        out = self.maxpool1(out)
        out = self.conv1_drop(out)

        # conv layer 2
        out = self.conv3(out)
        out = self.relu3(out)
        out = self.batch3(out)

        out = self.conv4(out)
        out = self.relu4(out)
        out = self.batch4(out)

        out = self.maxpool2(out)
        out = self.conv2_drop(out)

        # conv layer 3
        out = self.conv5(out)
        out = self.relu5(out)
        out = self.batch5(out)

        out = self.conv6(out)
        out = self.relu6(out)
        out = self.batch6(out)

        out = self.maxpool3(out)
        out = self.conv3_drop(out)
        # Flatten
        out = out.view(out.size(0), -1)

        # FC layer
        out = self.fc1(out)
        out = self.fc1_relu(out)
        out = self.dp1(out)

        out = self.fc2(out)

        return F.log_softmax(out, dim=1)

def weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

#instantiate
network = CNNModel()
network.to(device)
network.apply(weight_init)

#load model
model_path = "./model_taskA.pth"  #change this to "./model_taskA.pth" if testing for task A
network.load_state_dict(torch.load(model_path))
network.eval()


path = './test/'
data = pd.read_csv("./test/label.csv")
img_label = data['label'].tolist()
img_name = data['file_name'].tolist()
image_dir = "./test/image"
counter = 0
loss = 0
total = 200


class MyDataset(torch.utils.data.Dataset):  # Create Dataset
    def __init__(self, transform=torchvision.transforms.ToTensor(), target_transform=None):  # Initializer
        categories = ["meningioma_tumor", "glioma_tumor", "pituitary_tumor", "no_tumor"]
        imgs = []
        for i in range(len(img_name)):
            if (img_label == "meningioma_tumor") or (img_label == "glioma_tumor") or (
                    img_label == "pituitary_tumor"):
                to_append = 1  # 1 stands for tumor
            else:
                to_append = 0  # 0 stands for no-tumor
            imgs.append((img_name[i], to_append))  # append to imgs array
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):  # get item
        fn, label = self.imgs[index]  # fn = filename; label = tumor type
        img = Image.open(image_dir + '/' + fn).convert('L')  # load path
        img=img.resize((128,128))
        if self.transform is not None:
            img = self.transform(img)  # transform
        return img, label

    def __len__(self):
        return len(self.imgs)


start = time.time()
test_set = MyDataset()
test_loader = DataLoader(dataset=test_set, batch_size=64)
with torch.no_grad():
    for data, target in test_loader:
        output = network(data.to(device))  # forward
        # test_loss += F.nll_loss(output, target, size_average=False).item()
        loss += F.nll_loss(output, target.to(device), reduction='sum').item()
        pred = output.data.max(dim=1, keepdim=True)[1]  # take maximum
        counter += pred.eq(target.data.view_as(pred).to(device)).sum()  # validate correctness
end = time.time()
acc = counter/total
print('accuracy:',acc*100,'%')
print('time elapsed:',end-start, 'seconds')