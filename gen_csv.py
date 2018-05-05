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
import torch.nn.functional as F

test_pred = torch.load('test_prediction.pth')
test_prob = F.softmax(Variable(test_pred['prediction']), dim=1).data.numpy()
test_prob = test_prob.mean(axis=2)
test_predicted = np.argmax(test_prob, axis=1)

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

data_dir = 'dataset'
image_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform)
data_loader = torch.utils.data.DataLoader(image_dataset, batch_size=16, shuffle=False, num_workers=1)

IMAGE_SIZE = 224
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((IMAGE_SIZE + 20, IMAGE_SIZE + 20)),
        transforms.RandomCrop((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = 'dataset'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16,
                                             shuffle=True, num_workers=8)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes


img_list = []
for i in range(len(image_dataset.imgs)):
    path = image_dataset.imgs[i][0]
    idx = int(path.split('.')[0].split('/')[-1])
    img_list.append(idx)
    test_predicted[i] = class_names[test_predicted[i]]

import pandas as pd
sx = pd.read_csv('imaterialist-challenge-furniture-2018/sample_submission_randomlabel.csv')

for i in range(len(img_list)):
    sx.loc[img_list[i]-1,'predicted'] = test_predicted[i]

sx.to_csv('submission.csv', index=False)
