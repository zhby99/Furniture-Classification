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
from tqdm import tqdm

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

def get_model():
    print('[+] loading model... ', end='', flush=True)
    model = torchvision.models.densenet201(pretrained=True)
    model.classifier = nn.Linear(model.classifier.in_features, 128)
    if use_gpu:
        model.cuda()
    print('done')
    return model


data_transforms = []

data_transforms.append(transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]))
data_transforms.append(transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    HorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]))


data_dir = 'dataset'
data_loaders = []
for transform in data_transforms:
    image_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform)
    data_loader = torch.utils.data.DataLoader(image_dataset, batch_size=16, shuffle=False, num_workers=1)
    data_loaders.append(data_loader)


use_gpu = torch.cuda.is_available()


def predict(model, dataloader):
    all_outputs = []
    model.eval()

    pbar = tqdm(dataloader, total=len(dataloader))
    for inputs, labels in pbar:
        inputs = Variable(inputs, volatile=True)
        if use_gpu:
            inputs = inputs.cuda()

        outputs = model(inputs)
        all_outputs.append(outputs.data.cpu())

    all_outputs = torch.cat(all_outputs)
    if use_gpu:
        all_outputs = all_outputs.cuda()

    return all_outputs

def safe_stack_2array(a, b, dim=0):
    if a is None:
        return b
    return torch.stack((a, b), dim=dim)


def test_model(model):
    prediction = None
    for dataloader in data_loaders:
        px = predict(model, dataloader)
        prediction = safe_stack_2array(prediction, px, dim=-1)
    return prediction

model = get_model()
model.load_state_dict(torch.load('best_weight.pth'))
model.eval()
prediction = test_model(model)
data = {
    'prediction': prediction.cpu(),
}
torch.save(data, 'val_prediction.pth')
