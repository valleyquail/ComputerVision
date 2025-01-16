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
from typing import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from debian.debtags import output
from sympy.physics.units import boltzmann_constant
from torch.nn import ReLU6


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


def loss_fn(out, labels):
    ###############################################
    # Fill your loss function of choice here!
    ###############################################
    criterion = nn.CrossEntropyLoss()
    loss = criterion(out, labels)
    return loss


class ImageClassificationBase(nn.Module):
    def __init__(self):
        super(ImageClassificationBase, self).__init__()
        self.last_loss = np.inf

    def training_step(self, batch, device):
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)
        out = self(images)  # Generate predictions
        loss = loss_fn(out, labels)  # Calculate loss
        return loss

    def validation_step(self, batch, device):
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)
        out = self(images)  # Generate predictions
        loss = loss_fn(out, labels)  # Calculate loss
        acc = accuracy(out, labels)  # Calculate accuracy
        return {'loss': loss.detach(), 'acc': acc, 'preds': out, 'labels': labels}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_accs = [x['acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
        return {'loss': epoch_loss.item(), 'acc': epoch_acc.item()}

    def epoch_end(self, epoch, minibatch, result):
        if result['loss'] < self.last_loss:
            self.last_loss = result['loss']
            print("\nEpoch {}, Minibatch {}, loss: {:.4f}, acc: {:.4f}".format(epoch, minibatch, result['loss'],
                                                                               result['acc']))


class CIFAR10Model_Basic_Linear(ImageClassificationBase):
    def __init__(self, InputSize, OutputSize):
        super(CIFAR10Model_Basic_Linear, self).__init__()
        """
        Inputs:
        InputSize - Size of the Input
        OutputSize - Size of the Output
        """
        #############################
        # Fill your network initialization of choice here!
        #############################
        self.fc1 = nn.Linear(InputSize, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, OutputSize)
        for layer in [self.fc1, self.fc2, self.fc3, self.fc4]:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, xb):
        """
        Input:
        xb is a MiniBatch of the current image
        Outputs:
        out - output of the network
        """
        #############################
        # Fill your network structure of choice here!
        #############################
        out = xb.view(xb.shape[0], -1)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        out = F.relu(out)
        out = self.fc4(out)

        return out


class CIFAR10Model_Basic_CNN(ImageClassificationBase):
    def __init__(self, InputSize, OutputSize):
        super(CIFAR10Model_Basic_CNN, self).__init__()
        """
        Inputs:
        InputSize - Size of the Input
        OutputSize - Size of the Output
        """
        #############################
        # Fill your network initialization of choice here!
        #############################
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(65536, 128)
        self.fc2 = nn.Linear(128, OutputSize)

    def forward(self, xb):
        """
        Input:
        xb is a MiniBatch of the current image
        Outputs:
        out - output of the network
        """
        #############################
        # Fill your network structure of choice here!
        #############################

        out = self.conv1(xb)
        out = F.relu(out)

        out = self.conv2(out)
        out = F.relu(out)

        out = self.conv3(out)
        out = F.relu(out)

        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        return out


class CIFAR10Model_BN_CNN(ImageClassificationBase):
    def __init__(self, InputSize, OutputSize):
        super(CIFAR10Model_BN_CNN, self).__init__()
        """
        Inputs:
        InputSize - Size of the Input
        OutputSize - Size of the Output
        """
        #############################
        # Fill your network initialization of choice here!
        #############################
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(65536, 128)
        self.fc2 = nn.Linear(128, OutputSize)

    def forward(self, xb):
        """
        Input:
        xb is a MiniBatch of the current image
        Outputs:
        out - output of the network
        """
        #############################
        # Fill your network structure of choice here!
        #############################

        out = self.conv1(xb)
        out = F.relu(out)

        out = self.conv2(out)
        out = F.relu(out)

        out = self.conv3(out)
        out = F.relu(out)

        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        return out


def conv_block(in_channels, out_channels, stride=1, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class BasicBlock(nn.Module):
    """
    Basic ResNet block with two 3x3 convolutions and skip connection
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection (identity mapping)
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

#Assisted architecture with Claude, but implemented by myself
class CIFAR10Model_ResNet(ImageClassificationBase):
    def __init__(self, InChannels=3, OutputSize=10):
        super().__init__()

        # Initial layer
        self.conv1 = nn.Conv2d(InChannels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # ResNet layers
        self.layer1 = nn.Sequential(BasicBlock(64, 64),
                                    BasicBlock(64, 64))
        self.layer2 = nn.Sequential(BasicBlock(64, 128, stride=2),
                                    BasicBlock(128, 128))
        self.layer3 = nn.Sequential(BasicBlock(128, 256, stride=2),
                                    BasicBlock(256, 256))
        self.layer4 = nn.Sequential(BasicBlock(256, 512, stride=2),
                                    BasicBlock(512, 512))


        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(512, OutputSize)

        # Weight initialization
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x


# Referenced from https://medium.com/@atakanerdogan305/resnext-a-new-paradigm-in-image-processing-ee40425aea1f
class ResNeXtBlock(nn.Module):
    def __init__(self, in_channels, bottleneck_channels, out_channels, cardinality, stride=1):
        super(ResNeXtBlock, self).__init__()
        self.cardinality = cardinality
        self.conv1x1_1 = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1)
        self.conv3x3 = nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, stride=stride, padding=1,
                                 groups=cardinality)
        self.conv1x1_2 = nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.adapter = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        residual = x
        out = self.conv1x1_1(x)
        out = F.relu(out)
        out = self.conv3x3(out)
        out = F.relu(out)
        out = self.conv1x1_2(out)
        out = self.bn(out)
        out += self.adapter(residual)
        return out


class CIFAR10Model_ResNeXt(ImageClassificationBase):
    def __init__(self, cardinality, OutputSize=10):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(64)
        self.layer1 = nn.Sequential(
            ResNeXtBlock(self.in_channels, 128, 128, cardinality, stride=1),
            ResNeXtBlock(128, 128, 128, cardinality, stride=1),
            ResNeXtBlock(128, 128, 128, cardinality, stride=1)
        )
        self.layer2 = nn.Sequential(
            ResNeXtBlock(128, 256, 256, cardinality*2, stride=2),
            ResNeXtBlock(256, 256, 256, cardinality*2, stride=2),
            ResNeXtBlock(256, 256, 512, cardinality*2, stride=2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, OutputSize)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        out = self.bn(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.avgpool(out)
        out = out.flatten(1)
        out = self.fc(out)
        return out


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(DenseBlock, self).__init__()
        self.layers = []
        for i in range(num_layers):
            self.layers.append(nn.Sequential(
                nn.BatchNorm2d(in_channels + i * growth_rate),
                nn.ReLU(),
                nn.Conv2d(in_channels + i * growth_rate, growth_rate, kernel_size=3, padding=1)
            ))

        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        for layer in self.layers:
            out = layer(x)
            x = torch.cat([x, out], 1)
        return x

# Referenced from https://github.com/andreasveit/densenet-pytorch/blob/master/densenet.py
class CIFAR10Model_DenseNet(ImageClassificationBase):
    def __init__(self, InChannels, OutputSize, GrowthFactor):
        """
        Inputs:
        InputSize - Size of the Input
        OutputSize - Size of the Output
        """
        #############################
        # Fill your network initialization of choice here!
        #############################
        super(CIFAR10Model_DenseNet, self).__init__()
        self.conv1 = nn.Conv2d(InChannels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.dense1 = DenseBlock(64, GrowthFactor, 4)
        self.trans1 = nn.Sequential(
            nn.BatchNorm2d(64 + 4 * GrowthFactor),
            nn.ReLU(),
            nn.Conv2d(64 + 4 * GrowthFactor, 128, kernel_size=1),
            nn.AvgPool2d(2)
        )
        self.dense2 = DenseBlock(128, GrowthFactor, 4)
        self.trans2 = nn.Sequential(
            nn.BatchNorm2d(128 + 4 * GrowthFactor),
            nn.ReLU(),
            nn.Conv2d(128 + 4 * GrowthFactor, 256, kernel_size=1),
            nn.AvgPool2d(2)
        )
        self.dense3 = DenseBlock(256, GrowthFactor, 3)
        self.bn2 = nn.BatchNorm2d(256 + 3 * GrowthFactor)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256 + 3 * GrowthFactor, OutputSize)

    def forward(self, xb):
        """
        Input:
        xb is a MiniBatch of the current image
        Outputs:
        out - output of the network
        """
        #############################
        # Fill your network structure of choice here!
        #############################
        out = self.conv1(xb)
        out = F.relu(self.bn1(out))
        out = self.dense1(out)
        out = self.trans1(out)
        out = self.dense2(out)
        out = self.trans2(out)
        out = self.dense3(out)
        out = F.relu(self.bn2(out))
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out
