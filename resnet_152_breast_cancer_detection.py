# Breast Cancer Detection

#########################################################
# importing Libraries
#!pip install Pillow
#!pip install image
#!pip install pytorch torchvision
#!pip install gdown
# Pillow is the Python Image Library.
import PIL
# Django Applictin that provides cropping, resizing, thumbnailing, overlays and masking for images and videos.
from PIL import Image
# Provides Tensor computation (like numpy) with GPU acceleration and Deep Neural Networks built on a tape-based autograd system.
import torch
# Provides common image transformations
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms, models
import numpy as np
#import matplotlib.pyplot as plt
# Downloading a large file from Google Drive
import gdown
import os
import time
import copy

from collections import OrderedDict

########################################################
# Training on CPU or GPU?
# Check to see if CUDA is available.
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available. Training on CPU ...')
else:
    print('CUDA is available. Training on GPU ...')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

########################################################
# Download the data
output = 'breast_cancer_data_v1.zip'

# Complete dataset ~4GB
#url = 'https://drive.google.com/uc?id=1k2RHhOLHYv2mTLk0GE0SajjfvpziEHJI'

# mini dataset (10 malignant and 10 benign PNGs)
#url = 'https://drive.google.com/uc?id=12L3PE1YI-XOXdyuLNIe7cHu-JaoqMpW3'

#gdown.download(url, output, quiet=False)
#!tar xf {output}

# Organize the dataset
#x = %pwd   # find current directory
x = 'C:/users/Craig'
data_home = x + '/cancer_data_v1'
train_dir = data_home + '/train'
valid_dir = data_home + '/valid'
num_workers = 4
batch_size = 32

##########################################################
# Transforms, Augmentation, and Normalization
# Define transforms, data augmentation, and normalization
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

##########################################################
# Load Datasets wtih ImageFolder
# use ImageFolder to load the dataset
image_datasets = {x: datasets.ImageFolder(os.path.join(data_home, x), data_transforms[x])
    for x in ['train', 'valid']}

# Using the image datasets and the trainforms, define the dataloaders
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
    shuffle=True, num_workers=num_workers) for x in ['train', 'valid']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}

class_names = image_datasets['train'].classes

#############################################################
# Building and training the classifier
# Picking a pretrained model: Resnet-152
# We decided to used the pre-trained Resnet-152 to extract features for classification.

# Resnet-152 is a type of specialized neural network that helps to handle more
# sophisticated deep learning tasks and models. Resnet introduces a structure
# called residual learning unit to alleviate the degradation of deep neural networks.
# The structure is a feedforward network with a shortcut connection which adds new
# inputs into the network and generates new outputs. The main merit of this unit is
# that it produces better classification accuracy without increasing the complexity of the model

# Build and train your network

# Load resnet-152 pre-trained network
model = models.resnet152(pretrained=True)
# Freeze parameters so we don't backprop through them

for param in model.parameters():
    param.requires_grad = False
# Let's check the model architecture:
    #print(model)

# Define a new, untrained feed-forward network as a classifier, using ReLU activations
# Our input_size matches the in_features of pretrained model
# Creating the classifier ordered dictionary first
classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(2048, 512)),
                          ('relu', nn.ReLU()),
                          ('dropout1', nn.Dropout(p=0.5)),
                          ('fc2', nn.Linear(512, 2)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

# Replacing the pretrained model classifier with our classifier
model.fc = classifier

#Function to train the model
def train_model(model, criterion, optimizer, scheduler, num_epochs=10):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(1, num_epochs+1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / dataset_sizes[phase]
            #epoch_acc = running_corrects.double() / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best valid accuracy: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# Train a model with a pre-trained network
num_epochs = 1
if train_on_gpu:
    print ("Using GPU: "+ str(use_gpu))
    model = model.cuda()

# NLLLoss because our output is LogSoftmax
criterion = nn.NLLLoss()


# Adam optimizer with a learning rate
optimizer = optim.Adam(model.fc.parameters(), lr=0.005)
#optimizer = optim.SGD(model.fc.parameters(), lr = .0006, momentum=0.9)
# Decay LR by a factor of 0.1 every 5 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

model_ft = train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs=10)

