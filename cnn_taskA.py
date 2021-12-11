import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.image as image
import cv2
import numpy
import os
import pandas as pd

# cuda GPU
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
torch.cuda.empty_cache()

# hyper parameter
n_epochs = 10  #number of training epochs
batch_size_train = 240 #batch_size for training
batch_size_test = 1000 #batch_size for testing
learning_rate = 0.0001 # 0.01~0.0001 smaller means slower but more accurate
momentum = 0.5 # SGD algorithm speed multiplier, 0.5 = double; 0.9 = 10x speed
log_interval = 10
random_seed = 2
torch.manual_seed(random_seed)

# load data
data = pd.read_csv("./dataset/label.csv")
img_label = data['label'].tolist()
img_name = data['file_name'].tolist()
image_dir = "./dataset/image"
to_append = 0


class MyDataset(torch.utils.data.Dataset):  # Create Dataset
    def __init__(self, transform=torchvision.transforms.ToTensor(), target_transform=None):  # Initializer
        categories = ["meningioma_tumor", "glioma_tumor", "pituitary_tumor", "no_tumor"]
        imgs = []
        for i in range(len(img_name)):
            if (img_label == "meningioma_tumor") or (img_label == "glioma_tumor") or (
                    img_label == "pituitary_tumor"):
                to_append = 1    # 1 stands for tumor
            else:
                to_append = 0   # 0 stands for no-tumor
            imgs.append((img_name[i],to_append)) # append to imgs array
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


# split dataset for training and validation
all_data=MyDataset()
length = len(all_data)
train_size,validate_size=int(0.8*length),int(0.2*length)
train_set,validate_set=torch.utils.data.random_split(all_data,[train_size,validate_size])
print(train_set,validate_set)

# data loader

train_loader = DataLoader(dataset=train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=validate_set, batch_size=64)

print(train_loader.dataset)
print(test_loader.dataset)

# test with enumerate
examples = enumerate(test_loader)
# get batch
batch_idx, (example_data, example_targets) = next(examples)
# test target
print(example_targets)
print(example_data.shape)


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

# weight initializer


def weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')


# instantiate network
network = CNNModel()
network.to(device)
network.apply(weight_init)
# set up optimizer, SGD, Adam and RMSprop available for choosing
#optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
#optimizer = optim.Adam(network.parameters(), lr=learning_rate)
optimizer = optim.RMSprop(network.parameters(),lr=learning_rate,alpha=0.99,momentum = momentum)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True,
                                           threshold=0.00005, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)


# statistical data to be stored
train_losses = []
train_counter = []
train_acces = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]
test_acces = []


# training function
def train(epoch):
    network.train()  # train mode
    train_correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        # zero gradient
        optimizer.zero_grad()

        # forward calculation
        output = network(data.to(device))

        # loss calculation
        loss = F.nll_loss(output, target.to(device))

        # backward
        loss.backward()

        # optimize
        optimizer.step()
        # exp_lr_scheduler.step()

        train_pred = output.data.max(dim=1, keepdim=True)[1]  # choose maximum output

        train_correct += train_pred.eq(target.data.view_as(train_pred).to(device)).sum()  # validate correctness

        print('\r Number {}  Train Epoch: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                   100. * batch_idx / len(train_loader), loss.item()), end='')

        # log_interval = 10
        if batch_idx % log_interval == 0:
            # print(batch_idx)
            # print statistical data
            '''print('\r Number {}  Train Epoch: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()),end = '')
            '''
            # store loss for plotting
            train_losses.append(loss.item())
            # counter
            train_counter.append(
                (batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))
            '''
            torch.save(network.state_dict(), './model.pth')
            torch.save(optimizer.state_dict(), './optimizer.pth')
            '''

    train_acc = train_correct / len(train_loader.dataset)
    train_acces.append(train_acc.cpu().numpy().tolist())
    print('\tTrain Accuracy:{:.2f}%'.format(100. * train_acc))


# Test function
def test(epoch):
    network.eval()  # evaluation mode
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data.to(device))  # forward
            # test_loss += F.nll_loss(output, target, size_average=False).item()
            test_loss += F.nll_loss(output, target.to(device), reduction='sum').item()

            pred = output.data.max(dim=1, keepdim=True)[1]  # take maximum

            correct += pred.eq(target.data.view_as(pred).to(device)).sum()  # validate correctness
    acc = correct / len(test_loader.dataset)
    test_acces.append(acc.cpu().numpy().tolist())

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)  # record test_loss

    if test_acces[-1] >= max(test_acces):
        # save model after testing
        torch.save(network.state_dict(), './model_taskA.pth')

        # save optimizer after testing
        torch.save(optimizer.state_dict(), './optimizer_taskA.pth')

    # print statistical data
    '''
    print('\rTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)),end = '')
    '''
    print('\r Test set \033[1;31m{}\033[0m : Avg. loss: {:.4f}, Accuracy: {}/{}  \033[1;31m({:.2f}%)\033[0m\n' \
          .format(epoch, test_loss, correct, len(test_loader.dataset), 100. * acc), end='')

test(1)


###################################################
# training and testing
for epoch in range(1, n_epochs + 1):
    scheduler.step(test_acces[-1])
    train(epoch)
    test(epoch)
print('\n\033[1;31mThe network Max Avg Accuracy : {:.2f}%\033[0m'.format(100. * max(test_acces)))

# visualization
fig = plt.figure(figsize=(15,5))
ax1 = fig.add_subplot(121)
ax1.plot(train_counter, train_losses, color='blue')
plt.scatter(test_counter, test_losses, color='red')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.title('Train & Test Loss')
plt.xlabel('number of training examples seen')
plt.ylabel('negative log likelihood loss')

plt.subplot(122)

max_test_acces_epoch = test_acces.index(max(test_acces))
max_test_acces = round(max(test_acces),4)


plt.plot([epoch+1 for epoch in range(n_epochs) ], train_acces, color='blue')
plt.plot([epoch+1 for epoch in range(n_epochs) ], test_acces[1:], color='red')

plt.plot(max_test_acces_epoch,max_test_acces,'ko') #最大值点

show_max='  ['+str(max_test_acces_epoch )+' , '+str(max_test_acces)+']'
plt.annotate(show_max,xy=(max_test_acces_epoch,max_test_acces),
             xytext=(max_test_acces_epoch,max_test_acces))

plt.legend(['Train acc', 'Test acc'], loc='lower right')
plt.title('Train & Test Accuracy')
#plt.ylim(0.8, 1)
plt.xlabel('number of training epoch')
plt.ylabel('negative log likelihood acc')
plt.show()