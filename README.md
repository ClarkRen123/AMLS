# AMLS 

Author: Clark Ren 

SN: 18001986

Please download essential dataset file and test file from moodle. After downloading the two zip folders from moodle, unzip the zipped files in one working directory. E.g. C:\AMLS


After downloading the 8 python files, please put them in the same directory with the dataset folder and test folder. Below is a screenshot of how the directory looks like after you downloading and running the programs.

![Alt text](./view.png)

## supervised__A.py
Supervised learning agent for task A. When running the agent, you may be asked to input number of training files. This function is used to allow flexible debugging.
If you want to train for all files, just type 3000 and press enter. There's no validation in the interface so if wrong value is entered, program might crash.
Then, the interface will show a message to allow you choose supervised learning classifiers, please follow the instruction and type integer only to choose classifier.

After running the program, a pickle file called supervisedTask_A.p is generated. It stores the trained model. Re-running this program and changing classifier settings will overwrite the saved model.

## supervised_test_A.py
Testing program, need saved model from supervised__A.py and test folder to run.

## supervised_B.py
Supervised learning agent for task B. The instructions are the same as supervised__A.py. Please make sure not to enter wrong values. After running the program, a pickle file called supervisedTaskB.p is generated.

## supervised_test_A.py
Testing program, need saved model from supervised_B.py and test folder to run.

## cnn_taskA.py
CNN model used for task A. Installation of cuda toolkit is necessary to run the prorgam. Please make sure cuda is installed with correct version. After running the program, model and corresponding optimizer are saved as model_taskA.pth and optimizer_taskA.pth.

## cnn_test_A.py
Test program for deep learning in binary case. Cuda is necessary.

## cnn_train.py
CNN model used for task B. Installation of cuda toolkit is necessary to run the prorgam. After running the program, model and corresponding optimizer are saved as model_taskB.pth and optimizer_taskB.pth.

## cnn_test.py
Test program for deep learning in multi-class case. Cuda is necessary.
