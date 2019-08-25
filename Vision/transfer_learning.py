import warnings
warnings.filterwarnings("ignore")
import logging as lg
lg.basicConfig(level=lg.INFO, format='%(levelname)-8s: %(message)s')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from six.moves import cPickle as pickle
from  PIL import Image
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import time
import os
import argparse

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def imshow(img):
    img = img / 2 + 0.35     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),\
                                             torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                             ])
data_path = 'data/'
fpath = os.path.join(data_path)

# data = torchvision.datasets.MNIST('data/', download=False)
# print(data)
# exit(0)

train_data = torchvision.datasets.CIFAR10(root=fpath, train=True, transform=transform,\
                                            target_transform=None, download=False)

test_data = torchvision.datasets.CIFAR10(root=fpath, train=False, transform=transform, \
                                            target_transform=None, download=False)


train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True, sampler=None, batch_sampler=None, num_workers=0,
                             pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None)

test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True, sampler=None, batch_sampler=None, num_workers=0,
                             pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None)



classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')  # correct
# classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'truck', 'ship')    # incorrect

# resnet18 = models.resnet18(pretrained=True)
# alexnet = models.alexnet(pretrained=True) # https://download.pytorch.org/models/vgg16-397923af.pth
# vgg16 = models.vgg16(pretrained=True) # https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth

dataiter = iter(test_loader)
images, labels = dataiter.next()

net = models.resnet18(pretrained=True)

freeze_layers = True
n_class = len(classes)

if freeze_layers:
  for i, param in net.named_parameters():
    param.requires_grad = False

num_ftrs = net.fc.in_features
net.fc = nn.Linear(num_ftrs, n_class)
net.fc1 = nn.Softmax(n_class)

# Stage-2 , Freeze all the layers till "Conv2d_4a_3*3"
ct = []
for name, child in net.named_children():
    if "Conv2d_4a_3x3" in ct:
        for params in child.parameters():
            params.requires_grad = True
    ct.append(name)

# exit(0)

optimizer = torch.optim.Adam(net.parameters(), lr=0.005)
criterion = nn.CrossEntropyLoss()
# Train
for epoch in range(1):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 200 == 199:    # print every 200 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0

print('Finished Training')


# Accuracy
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

