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

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def get_model():
    print('[+] loading model... ', end='', flush=True)
    model = torchvision.models.densenet201(pretrained=True)
    if use_gpu:
        model.cuda()
    print('done')
    return model


data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


data_dir = 'dataset'
image_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), data_transform)
dataloader = torch.utils.data.DataLoader(image_datasets['test'], batch_size=16,
                                             shuffle=True, num_workers=1)
dataset_size= len(image_dataset)
# class_names = image_datasets['train'].classes

use_gpu = torch.cuda.is_available()

def test_model(model):
    model = get_model()
    model.load_state_dict(torch.load('best_weight.pth'))
    model.eval()
    for data in dataloader:
        
