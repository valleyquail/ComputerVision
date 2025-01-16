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

def ConfusionMatrix(LabelsTrue, LabelsPred):
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
    plt.title('Confusion Matrix for DenseNet Test')
    plt.ylabel('Predicted label')
    plt.xlabel('True Label\n' + 'Accuracy: ' +str(accuracy))

    plt.colorbar()
    plt.tight_layout()
    plt.show()

    # Print the class-numbers for easy reference.
    class_numbers = [" ({0})".format(i) for i in range(10)]
    print("".join(class_numbers))

    print('Accuracy: '+ str(accuracy), '%')


def TestOperation(ImageSize, ModelPath, TestSet, LabelsPathPred):
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
    model = CIFAR10Model_DenseNet(3,10, 24)
    
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
    Parser.add_argument('--ModelPath', dest='ModelPath', default='./../Checkpoints/DenseNet/best_DenseNet_model.ckpt', help='Path to load latest model from, Default:ModelPath')
    Parser.add_argument('--LabelsPath', dest='LabelsPath', default='./TxtFiles/LabelsTrain.txt', help='Path of labels file, Default:./TxtFiles/LabelsTest.txt')
    Args = Parser.parse_args()
    ModelPath = Args.ModelPath
    LabelsPath = Args.LabelsPath
    # TestSet = CIFAR10(root='data/', train=False)
    TestSet = CIFAR10(root='./data', train=True, download=False)

    # Setup all needed parameters including file reading
    ImageSize = SetupAll()

    # Define PlaceHolder variables for Predicted output
    LabelsPathPred = './TxtFiles/DenseNet/PredOut.txt' # Path to save predicted labels

    TestOperation(ImageSize, ModelPath, TestSet, LabelsPathPred)

    # Plot Confusion Matrix
    LabelsTrue, LabelsPred = ReadLabels(LabelsPath, LabelsPathPred)
    ConfusionMatrix(LabelsTrue, LabelsPred)
     
if __name__ == '__main__':
    main()
 
