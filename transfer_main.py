from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

import torchvision.transforms.functional as F

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

class HorizontalFlip(object):
    """Horizontally flip the given PIL Image."""

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be flipped.
        Returns:
            PIL Image: Flipped image.
        """
        return F.hflip(img)

# Data augmentation and normalization for training
# Just normalization for validation
IMAGE_SIZE = 224
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'train_flip': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        HorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val_flip': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        HorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = 'dataset'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x.split('_')[0]), data_transforms[x])
                  for x in ['train', 'train_flip', 'val', 'val_flip']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16,
                                             shuffle=True, num_workers=8)
              for x in ['train', 'train_flip', 'val', 'val_flip']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'train_flip', 'val', 'val_flip']}
class_names = image_datasets['train'].classes

use_gpu = torch.cuda.is_available()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(model, criterion, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    loss_list = []
    acc_list = []
    lr = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        if epoch == 1:
            lr = 0.00003
        if epoch == 0:
            lr = 0.001
            optimizer = torch.optim.Adam(model.classifier.parameters(), lr=lr)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)

        # Each epoch has a training and validation phase
        for phase in ['train','train_flip', 'val', 'val_flip']:
            # if phase == 'train':
            # scheduler.step()
            model.train()  # Set model to training mode
            # else:
            #     model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                with torch.set_grad_enabled(True):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    # if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            loss_list.append(epoch_loss)
            acc_list.append(epoch_acc)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # # deep copy the model
            # if phase == 'val' and epoch_acc > best_acc:
            #     best_acc = epoch_acc
            #     best_model_wts = copy.deepcopy(model.state_dict())


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    best_model_wts = copy.deepcopy(model.state_dict())
    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(best_model_wts, 'best_weight.pth')
    return model

model_conv = torchvision.models.densenet201(pretrained=True)
for param in model_conv.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
model_conv.classifier = nn.Linear(model_conv.classifier.in_features, 128)


if use_gpu:
    model_conv = model_conv.cuda()

criterion = nn.CrossEntropyLoss().cuda()

# Observe that only parameters of final layer are being optimized as
# opoosed to before.
# optimizer_conv = optim.SGD(model_conv.classifier.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=5, gamma=0.1)

model_conv = train_model(model_conv, criterion, num_epochs=20)
