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
# termcolor, do (pip install termcolor)


import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torch.optim import AdamW
from torchvision.datasets import CIFAR10
import cv2
import sys
import os
import numpy as np
import random
import skimage
import PIL
import os
import glob
import random
from skimage import data, exposure
import matplotlib.pyplot as plt
import time
from torchvision.transforms import ToTensor, Normalize
import argparse
import shutil
import string
from termcolor import colored, cprint
import math as m
from tqdm.auto import tqdm
import imgutils as iu
from Network.Network import *
from Misc.MiscUtils import *
from Misc.DataUtils import *
from Phase2.Code import Test
from Phase2.Code.Network.Network import CIFAR10Model_Basic_Linear
from Phase2.Code.Test import ReadImages, TestOperation

# Don't generate pyc codes
sys.dont_write_bytecode = True


def GenerateBatch(TrainSet, TrainLabels, ImageSize, MiniBatchSize):
    """
    Inputs:
    TrainSet - Variable with Subfolder paths to train files
    NOTE that Train can be replaced by Val/Test for generating batch corresponding to validation (held-out testing in this case)/testing
    TrainLabels - Labels corresponding to Train
    NOTE that TrainLabels can be replaced by Val/TestLabels for generating batch corresponding to validation (held-out testing in this case)/testing
    ImageSize is the Size of the Image
    MiniBatchSize is the size of the MiniBatch

    Outputs:
    I1Batch - Batch of images
    LabelBatch - Batch of one-hot encoded labels
    """
    I1Batch = []
    LabelBatch = []

    ImageNum = 0
    while ImageNum < MiniBatchSize:
        # Generate random image
        RandIdx = random.randint(0, len(TrainSet) - 1)

        ImageNum += 1

        I1, Label = TrainSet[RandIdx]

        ##########################################################
        # Add any standardization or data augmentation here!
        ##########################################################
        # Normalize the images to range [-1, 1]
        I1 = Normalize((0.5, 0.5, 0.5), (1, 1, 1))(I1)
        I1 = torchvision.transforms.RandomHorizontalFlip()(I1)
        # Append All Images and Mask
        I1Batch.append(I1)
        LabelBatch.append(torch.tensor(Label))

    return torch.stack(I1Batch), torch.stack(LabelBatch)


def PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile):
    """
    Prints all stats with all arguments
    """
    print('Number of Epochs Training will run for ' + str(NumEpochs))
    print('Factor of reduction in training data is ' + str(DivTrain))
    print('Mini Batch Size ' + str(MiniBatchSize))
    print('Number of Training Images ' + str(NumTrainSamples))
    if LatestFile is not None:
        print('Loading latest checkpoint with the name ' + LatestFile)


def TrainOperation(TrainLabels, NumTrainSamples, ImageSize,
                   NumEpochs, MiniBatchSize, SaveCheckPoint, CheckPointPath,
                   DivTrain, LatestFile, TrainSet, TestSet, LogsPath, LabelsPath, LabelsPredPath, ModelName):
    """
    Inputs:
    TrainLabels - Labels corresponding to Train/Test
    NumTrainSamples - length(Train)
    ImageSize - Size of the image
    NumEpochs - Number of passes through the Train data
    MiniBatchSize is the size of the MiniBatch
    SaveCheckPoint - Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
    CheckPointPath - Path to save checkpoints/model
    DivTrain - Divide the data by this number for Epoch calculation, use if you have a lot of dataor for debugging code
    LatestFile - Latest checkpointfile to continue training
    TrainSet - The training dataset
    TestSet - The test dataset
    LogsPath - Path to save Tensorboard Logs
    Outputs:
    Saves Trained network in CheckPointPath and Logs to LogsPath
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device is ' + str(device))
    torch.cuda.empty_cache()

    model = None
    LearningRate = 0.01
    Optimizer = None
    Scheduler = None
    match ModelName:
        case 'Basic_Linear':
            model = CIFAR10Model_Basic_Linear(InputSize=3 * 32 * 32, OutputSize=10)
            Optimizer = torch.optim.Adam(model.parameters(), lr=LearningRate)
            Scheduler = torch.optim.lr_scheduler.StepLR(Optimizer, step_size=3, gamma=0.25)
        case 'BN_CNN':
            model = CIFAR10Model_BN_CNN(InputSize=3 * 32 * 32, OutputSize=10)
            # NumEpochs = 12
            # LearningRate = 0.01
            # MiniBatchSize = 128
            Optimizer = torch.optim.Adam(model.parameters(), lr=LearningRate)
            Scheduler = torch.optim.lr_scheduler.StepLR(Optimizer, step_size=3, gamma=0.25)
        case 'ResNet':
            model = CIFAR10Model_ResNet(InChannels=3, OutputSize=10)
            # NumEpochs = 12
            # LearningRate = 0.01
            # MiniBatchSize = 200
            Optimizer = torch.optim.Adam(model.parameters(), lr=LearningRate, weight_decay=0.0001)
            Scheduler = torch.optim.lr_scheduler.StepLR(Optimizer, step_size=3, gamma=0.25)
        case 'ResNeXt':
            model = CIFAR10Model_ResNeXt(32, OutputSize=10)
            # NumEpochs = 8
            # LearningRate = 0.1
            # MiniBatchSize = 80
            Optimizer = torch.optim.SGD(model.parameters(), lr=LearningRate)
            Scheduler = torch.optim.lr_scheduler.StepLR(Optimizer, step_size=3, gamma=0.5)
        case 'DenseNet':
            # GrowthFactor = 24
            # LearningRate = 0.1
            # MiniBatchSize = 128
            model = CIFAR10Model_DenseNet(InChannels=3, OutputSize=10, GrowthFactor=GrowthFactor)
            Optimizer = torch.optim.SGD(model.parameters(), lr=LearningRate, weight_decay=0.00001)
            Scheduler = torch.optim.lr_scheduler.StepLR(Optimizer, step_size=3, gamma=0.25)
        case _:
            print('Model not found')

    ###############################################
    # Fill your optimizer of choice here!
    ###############################################

    # Tensorboard
    # Create a summary to monitor loss tensor
    Writer = SummaryWriter(LogsPath)

    if LatestFile is not None:
        CheckPoint = torch.load(CheckPointPath + LatestFile + '.ckpt')
        # Extract only numbers from the name
        StartEpoch = int(''.join(c for c in LatestFile.split('a')[0] if c.isdigit()))
        model.load_state_dict(CheckPoint['model_state_dict'])
        print('Loaded latest checkpoint with the name ' + LatestFile + '....')
    else:
        StartEpoch = 0
        print('New model initialized....')

    print(f"Training the {ModelName} model")
    ###############################################
    # Send the model to device
    model.to(device)
    ###############################################
    # Prediction output data saver
    ###############################################
    os.makedirs(LabelsPredPath.replace('PredOut.txt', ''), exist_ok=True)
    OutSaveT = open(LabelsPredPath, 'w')

    ##########################################################
    # Read the ground truth labels
    ##########################################################
    LabelsTest = None
    if (not (os.path.isfile(LabelsPath))):
        print('ERROR: Test Labels do not exist in ' + LabelsPath)
        sys.exit()
    else:
        LabelsTest = open(LabelsPath, 'r')
        LabelsTest = LabelsTest.read()
        LabelsTest = map(float, LabelsTest.split())
    LabelsTest = np.array(list(LabelsTest))
    ##########################################################
    # Data collection arrays
    ##########################################################
    train_accuracy_dt = []
    train_loss_dt = []
    validation_accuracy_dt = []
    validation_loss_dt = []
    best_accuracy = 0
    early_stop = 0
    for Epochs in tqdm(range(StartEpoch, NumEpochs)):
        NumIterationsPerEpoch = int(NumTrainSamples / MiniBatchSize / DivTrain)
        LossThisBatch = None
        loss_train = 0
        acc_train = 0
        for PerEpochCounter in tqdm(range(NumIterationsPerEpoch)):

            Batch = GenerateBatch(TrainSet, TrainLabels, ImageSize, MiniBatchSize)

            # Predict output with forward pass
            LossThisBatch = model.training_step(Batch, device)
            Optimizer.zero_grad()
            LossThisBatch.backward()
            Optimizer.step()


            # Save checkpoint every some SaveCheckPoint's iterations
            if PerEpochCounter % SaveCheckPoint == 0:
                # Save the Model learnt in this epoch
                SaveName = CheckPointPath + str(Epochs) + 'a' + str(PerEpochCounter) + '_' + ModelName + '_model.ckpt'

                torch.save({'epoch': Epochs, 'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': Optimizer.state_dict(), 'loss': LossThisBatch}, SaveName)
                print('\n' + SaveName + ' Model Saved...')

            result = model.validation_step(Batch, device)
            model.epoch_end(Epochs, PerEpochCounter, result)
            loss_train += result['loss'].item()
            acc_train += result['acc'].item()
            # Tensorboard
            Writer.add_scalar('LossEveryIter', result["loss"], Epochs * NumIterationsPerEpoch + PerEpochCounter)
            Writer.add_scalar('Accuracy', result["acc"], Epochs * NumIterationsPerEpoch + PerEpochCounter)
            # If you don't flush the tensorboard doesn't update until a lot of iterations!
            Writer.flush()
        train_accuracy_dt.append(acc_train / NumIterationsPerEpoch)
        train_loss_dt.append(loss_train / NumIterationsPerEpoch)
        # Save model every epoch
        SaveName = CheckPointPath + str(Epochs) + '_' + ModelName + '_model.ckpt'
        torch.save(
            {'epoch': Epochs, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': Optimizer.state_dict(),
             'loss': LossThisBatch}, SaveName)
        print('\n' + SaveName + ' Model Saved...')

        ##########################################################
        # Evaluate the model on the test dataset
        # Collect the accuracy and loss of the validation set for plotting
        ##########################################################
        # validation_loss = 0
        predictions = []
        outputs = []
        print('Evaluating the model on the test dataset')
        model.eval()

        for count in tqdm(range(len(TestSet))):
            with torch.no_grad():
                Img, Label = TestSet[count]
                Img, _ = ReadImages(Img)
                Img = torch.tensor(Img)
                Label = torch.tensor([Label])
                output = model.validation_step((Img, Label), device)
                outputs.append(output)

                PredT = torch.argmax(output['preds']).item()
                predictions.append(PredT)
            OutSaveT.write(str(PredT) + '\n')
        model.train()
        result = model.validation_epoch_end(outputs)
        if result['acc'] > best_accuracy:
            best_accuracy = result['acc']
            torch.save({'epoch': Epochs, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': Optimizer.state_dict(), 'loss': LossThisBatch},
                       CheckPointPath + 'best_' + ModelName + '_model.ckpt')
            early_stop = 0
            print('Best model saved at epoch:', Epochs)
        early_stop += 1
        validation_accuracy_dt.append(result['acc'])
        validation_loss_dt.append(result['loss'])
        print('Validation Accuracy: ', validation_accuracy_dt[-1] * 100, "%")
        print('Validation Loss: ', validation_loss_dt[-1])
        Scheduler.step()
        if early_stop == 2:
            print('Early stopping at epoch:', Epochs)
            break
    OutSaveT.close()
    Writer.close()
    print('Training Done!')

    plt.plot(train_accuracy_dt)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'{ModelName} Training Accuracy')
    plt.show()

    plt.plot(train_loss_dt)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'{ModelName} Training Loss')
    plt.show()

    plt.plot(validation_accuracy_dt)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'{ModelName} Validation Accuracy')
    plt.show()

    plt.plot(validation_loss_dt)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'{ModelName} Validation Loss')
    plt.show()


train = True


def main():
    """
    Inputs:
    None
    Outputs:
    Runs the Training and testing code based on the Flag
    """
    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--ModelName', default='Basic_Linear', help='Model Name, Default:bn_cnn')
    Parser.add_argument('--CheckPointPath', default='../Checkpoints/',
                        help='Path to save Checkpoints, Default: ../Checkpoints/')
    Parser.add_argument('--NumEpochs', type=int, default=12, help='Number of Epochs to Train for, Default:50')
    Parser.add_argument('--DivTrain', type=int, default=1, help='Factor to reduce Train data by per epoch, Default:1')
    Parser.add_argument('--MiniBatchSize', type=int, default=128, help='Size of the MiniBatch to use, Default:1')
    Parser.add_argument('--LoadCheckPoint', type=int, default=0,
                        help='Load Model from latest Checkpoint from CheckPointsPath?, Default:0')
    Parser.add_argument('--LogsPath', default='Logs/', help='Path to save Logs for Tensorboard, Default=Logs/')
    Parser.add_argument('--LabelsPath', dest='LabelsPath', default='./TxtFiles/',
                        help='Path of labels file, Default:./TxtFiles/')
    TrainSet = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=False, transform=ToTensor())
    TestSet = CIFAR10(root='data/', train=False)

    Args = Parser.parse_args()
    ModelName = Args.ModelName
    NumEpochs = Args.NumEpochs
    DivTrain = float(Args.DivTrain)
    MiniBatchSize = Args.MiniBatchSize
    LoadCheckPoint = Args.LoadCheckPoint
    CheckPointPath = Args.CheckPointPath
    LogsPath = Args.LogsPath
    LabelsPath = Args.LabelsPath

    CheckPointPath = CheckPointPath + Args.ModelName + '/'
    LogsPath = LogsPath + Args.ModelName + '/'
    LabelsPredPath = LabelsPath + Args.ModelName + '/' + 'PredOut.txt'
    LabelsPath = LabelsPath + 'LabelsTest.txt'

    # Setup all needed parameters including file reading
    DirNamesTrain, SaveCheckPoint, ImageSize, NumTrainSamples, TrainLabels, NumClasses = SetupAll("./", CheckPointPath)

    # Find Latest Checkpoint File
    if LoadCheckPoint == 1:
        LatestFile = FindLatestModel(CheckPointPath)
        files = os.listdir(CheckPointPath)
        for file in files:
            if file.find('best') >= 0:
                LatestFile = file.replace('.ckpt.index', '')
                print('Loading best model:', file)
    else:
        LatestFile = None

    # Pretty print stats
    PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile)

    if train:
        TrainOperation(TrainLabels, NumTrainSamples, ImageSize,
                       NumEpochs, MiniBatchSize, SaveCheckPoint, CheckPointPath,
                       DivTrain, LatestFile, TrainSet, TestSet, LogsPath, LabelsPath, LabelsPredPath, ModelName)


    BestModelPath = '../Checkpoints/' + ModelName + '/best_' + ModelName + '_model.ckpt'

    TestOperation(3 * 32 * 32, BestModelPath, TestSet, LabelsPredPath)


if __name__ == '__main__':
    main()
