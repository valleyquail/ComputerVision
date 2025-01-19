#!/usr/bin/env python3

"""
RBE/CS549 Spring 2022: Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code

Colab file can be found at:
    https://colab.research.google.com/drive/1FUByhYCYAfpl8J9VxMQ1DcfITpY8qgsF

Author(s): 
Prof. Nitin J. Sanket (nsanket@wpi.edu), Lening Li (lli4@wpi.edu), Gejji, Vaishnavi Vivek (vgejji@wpi.edu)
Robotics Engineering Department,
Worcester Polytechnic Institute

Code adapted from CMSC733 at the University of Maryland, College Park.
"""


# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (apt install python-skimage)

import cv2
import os
import sys
import glob
import random

import torchvision
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
import numpy as np
import time

from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, Normalize
import argparse
import shutil
import string
import math as m
from sklearn.metrics import confusion_matrix
from tqdm.auto import tqdm
import torch
from Network.Network import *
from Misc.MiscUtils import *
from Misc.DataUtils import *
from Phase1.Code.Plotting import plot_all

# Don't generate pyc codes
sys.dont_write_bytecode = True

def SetupAll():
    """
    Outputs:
    ImageSize - Size of the Image
    """   
    # Image Input Shape
    ImageSize = [32, 32, 3]

    return ImageSize

def StandardizeInputs(Img):
    ##########################################################################
    # Add any standardization or cropping/resizing if used in Training here!
    ##########################################################################
    Img = Normalize((0.5,0.5,0.5),(1,1,1))(ToTensor()(Img))
    return Img.numpy()
    
def ReadImages(Img):
    """
    Outputs:
    I1Combined - I1 image after any standardization and/or cropping/resizing to ImageSize
    I1 - Original I1 image for visualization purposes only
    """    
    I1 = Img
    
    if(I1 is None):
        # OpenCV returns empty list if image is not read! 
        print('ERROR: Image I1 cannot be read')
        sys.exit()
        
    I1S = StandardizeInputs(I1)

    I1Combined = np.expand_dims(I1S, axis=0)

    return I1Combined, I1
                

def Accuracy(Pred, GT):
    """
    Inputs: 
    Pred are the predicted labels
    GT are the ground truth labels
      Outputs:
    Accuracy in percentage
    """
    return (np.sum(np.array(Pred)==np.array(GT))*100.0/len(Pred))

def ReadLabels(LabelsPathTest, LabelsPathPred):
    if(not (os.path.isfile(LabelsPathTest))):
        print('ERROR: Test Labels do not exist in '+LabelsPathTest)
        sys.exit()
    else:
        LabelTest = open(LabelsPathTest, 'r')
        LabelTest = LabelTest.read()
        LabelTest = map(float, LabelTest.split())

    if(not (os.path.isfile(LabelsPathPred))):
        print('ERROR: Pred Labels do not exist in '+LabelsPathPred)
        sys.exit()
    else:
        LabelPred = open(LabelsPathPred, 'r')
        LabelPred = LabelPred.read()
        LabelPred = map(float, LabelPred.split())
        
    return LabelTest, LabelPred

def ConfusionMatrix(LabelsTrue, LabelsPred, ModelName, TestType):
    """
    LabelsTrue - True labels
    LabelsPred - Predicted labels
    """

    # Get the confusion matrix using sklearn.
    LabelsTrue, LabelsPred = list(LabelsTrue), list(LabelsPred)
    cm = confusion_matrix(y_true=LabelsTrue,  # True class for test-set.
                          y_pred=LabelsPred)  # Predicted class.

    # Print the confusion matrix as text.
    for i in range(10):
        print(str(cm[i, :]) + ' ({0})'.format(i))
    accuracy = Accuracy(LabelsPred, LabelsTrue)
    plt.imshow(cm, interpolation='nearest')
    plt.title('Confusion Matrix for ' + ModelName +' ' + TestType)
    plt.ylabel('Predicted label')
    plt.xlabel('True Label\n' + 'Accuracy: ' +str(accuracy))

    plt.colorbar()
    plt.tight_layout()
    plt.show()

    # Print the class-numbers for easy reference.
    class_numbers = [" ({0})".format(i) for i in range(10)]
    print("".join(class_numbers))

    print('Accuracy: '+ str(accuracy), '%')


def log_inference_times(ModelName, ModelPath, TestSet):
    print('Running inference time for model: ', ModelName)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device is ' + str(device))

    model = None
    match ModelName:
        case 'Basic_Linear':
            model = CIFAR10Model_Basic_Linear(InputSize=3 * 32 * 32, OutputSize=10)
        case 'BN_CNN':
            model = CIFAR10Model_BN_CNN(InputSize=3 * 32 * 32, OutputSize=10)
        case 'ResNet':
            model = CIFAR10Model_ResNet(InChannels=3, OutputSize=10)
        case 'ResNeXt':
            model = CIFAR10Model_ResNeXt(32, OutputSize=10)
        case 'DenseNet':
            GrowthFactor = 24
            model = CIFAR10Model_DenseNet(InChannels=3, OutputSize=10, GrowthFactor=GrowthFactor)
        case _:
            print('Model not found')
    ModelPath += ModelName + '/' + 'best_' + ModelName + '_model.ckpt'
    CheckPoint = torch.load(ModelPath)
    model.load_state_dict(CheckPoint['model_state_dict'])
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print('Number of parameters in this model are %d ' % num_params)
    model.to(device)
    model.eval()
    start_time = tic()
    for i in range(100):
        Img, Label = TestSet[i]
        Img, ImgOrg = ReadImages(Img)
        Img = torch.tensor(Img).to(device)
        _ = model(Img)
    print('Average inference time for one image is: ', toc(start_time) / 100)



def TestOperation(ImageSize, ModelPath, TestSet, LabelsPathPred, ModelName):
    """
    Inputs: 
    ImageSize is the size of the image
    ModelPath - Path to load trained model from
    TestSet - The test dataset
    LabelsPathPred - Path to save predictions
    Outputs:
    Predictions written to /content/data/TxtFiles/PredOut.txt
    """
    # Predict output with forward pass, MiniBatchSize for Test is 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device is ' + str(device))
    model = None

    match ModelName:
        case 'Basic_Linear':
            model = CIFAR10Model_Basic_Linear(InputSize=3 * 32 * 32, OutputSize=10)
        case 'BN_CNN':
            model = CIFAR10Model_BN_CNN(InputSize=3 * 32 * 32, OutputSize=10)
        case 'ResNet':
            model = CIFAR10Model_ResNet(InChannels=3, OutputSize=10)
        case 'ResNeXt':
            model = CIFAR10Model_ResNeXt(32, OutputSize=10)
        case 'DenseNet':
            GrowthFactor = 24
            model = CIFAR10Model_DenseNet(InChannels=3, OutputSize=10, GrowthFactor=GrowthFactor)
        case _:
            print('Model not found')



    
    CheckPoint = torch.load(ModelPath)
    model.load_state_dict(CheckPoint['model_state_dict'])
    print('Number of parameters in this model are %d ' % len(model.state_dict().items()))
    model.to(device)
    OutSaveT = open(LabelsPathPred, 'w')
    model.eval()
    for count in tqdm(range(len(TestSet))): 
        Img, Label = TestSet[count]
        Img, ImgOrg = ReadImages(Img)
        Img = torch.tensor(Img).to(device)
        PredT = torch.argmax(model(Img)).item()

        OutSaveT.write(str(PredT)+'\n')
    OutSaveT.close()

BasePath = "../CIFAR10/Test/"
def main():
    """
    Inputs: 
    None
    Outputs:
    Prints out the confusion matrix with accuracy
    """

    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()

    Parser.add_argument('--ModelName', default='Basic_Linear', help='Model Name, Default:Basic_Linear')
    Parser.add_argument('--ModelPath', dest='ModelPath', default='./../Checkpoints/', help='Path to load latest model from, Default:ModelPath')
    Parser.add_argument('--LabelsTrainPath', dest='LabelsTrainPath', default='./TxtFiles/LabelsTrain.txt',
                        help='Path of labels file, Default:./TxtFiles/LabelsTest.txt')
    Parser.add_argument('--LabelsTestPath', dest='LabelsTestPath', default='./TxtFiles/LabelsTest.txt',
                        help='Path of labels file, Default:./TxtFiles/LabelsTest.txt')
    Parser.add_argument('--RunInference', dest='RunInference', default='True', help='Run Inference, Default:False')
    Args = Parser.parse_args()
    ModelPath = Args.ModelPath
    LabelsTrainPath = Args.LabelsTrainPath
    LabelsTestPath = Args.LabelsTestPath
    ModelName = Args.ModelName
    ModelPathFull = ModelPath + ModelName + '/' + 'best_' + ModelName + '_model.ckpt'

    TestSet = CIFAR10(root='data/', train=False)
    TrainSet = CIFAR10(root='./data', train=True, download=False)

    # Setup all needed parameters including file reading
    ImageSize = SetupAll()

    # Define PlaceHolder variables for Predicted output
    LabelsPathPred = './TxtFiles/' + ModelName + '/PredOut.txt' # Path to save predicted labels

    if (Args.RunInference == 'True'):
        log_inference_times(ModelName, ModelPath, TestSet)
    return
    # Test Validation
    TestOperation(ImageSize, ModelPathFull, TestSet, LabelsPathPred, ModelName)
    # Plot Confusion Matrix
    LabelsTrue, LabelsPred = ReadLabels(LabelsTestPath, LabelsPathPred)
    ConfusionMatrix(LabelsTrue, LabelsPred, ModelName, 'Validation')

    #Train Validation
    TestOperation(ImageSize, ModelPathFull, TrainSet, LabelsPathPred, ModelName)
    # Plot Confusion Matrix
    LabelsTrue, LabelsPred = ReadLabels(LabelsTrainPath, LabelsPathPred)
    ConfusionMatrix(LabelsTrue, LabelsPred, ModelName, 'Train')

    plot_all()

     
if __name__ == '__main__':
    main()
 
