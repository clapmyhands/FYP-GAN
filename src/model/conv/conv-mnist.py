# coding: utf-8

import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import time
import visdom
import numpy as np

from torchvision import datasets, transforms
from torch.autograd import Variable

sys.path.append(os.path.join(os.path.dirname(__file__), "..",".."))
from util.helper import initializeWeight
from definitions import DATA_DIR

# dtype might be useful in transform with .type(dtype)
MNIST_DSET = os.path.join(DATA_DIR, "data_mnist")
train_dataset = datasets.MNIST(MNIST_DSET, train=True,
                               transform=transforms.ToTensor(),
                               download=True)
test_dataset = datasets.MNIST(MNIST_DSET, train=False,
                              transform=transforms.ToTensor(),
                              download=True)
train_size = len(train_dataset)
test_size = len(test_dataset)

batch_size = 64
train_loader = data.DataLoader(train_dataset, batch_size=batch_size,
                               shuffle=True, num_workers=4)
test_loader = data.DataLoader(test_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=4)

class convNet(nn.Module):
    def __init__(self):
        super(convNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.Dropout2d(),
            nn.MaxPool2d(2),
            nn.LeakyReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(10, 20, kernel_size=5),
            nn.Dropout2d(),
            nn.MaxPool2d(2),
            nn.LeakyReLU()
        )
        self.fc1 = nn.Linear(20*4*4, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, data):    # (?, 1, 28, 28)
        x = self.conv1(data)    # (?, 10, 24, 24) > (?, 10, 12, 12)
        x = self.conv2(x)       # (?, 10, 8, 8) > (?, 20, 4, 4)
        x = x.view(-1, 320)     # (?, 320)
        x = F.dropout(self.fc1(x), training=self.training)  # (?, 100)
        x = F.leaky_relu(x)     # (?, 100)
        x = self.fc2(x)         # (?, 10)
        return x


vis = visdom.Visdom()

# TODO: try out if dtype can be used instead of .cuda()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

epochs = 400
learning_rate = 1e-3

net = convNet()
net.apply(initializeWeight)
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss()

if torch.cuda.is_available():
    net = net.cuda()
    criterion = criterion.cuda()

def train(epoch):
    epoch_start = time.time()
    net.train()
    cost = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()

        pred = net(data)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch+1, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))
        cost += loss.data[0]*len(data)
    # TODO: move timer to wrapper
    print("time for training epoch: {}".format(time.time()-epoch_start))
    return cost/train_size


def test(epoch):
    net.eval()
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data), target
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()
        output = net(data)
        _, pred = torch.max(output.data, 1)
        correct += (pred == target).sum()
    print("Test accuracy for epoch {} : {}".format(epoch+1, 100*correct/test_size))
    return 100*correct/test_size


for epoch in range(0, epochs):
    cost = train(epoch)
    accuracy = test(epoch)
    # TODO: generalize and move this to helper with param cost/accuracy, epoch, train/test flag
    if(epoch == 0):
        train_cost = vis.line(X=np.array([epoch+1]), Y=np.array([cost]), opts=dict(
            title='Train Cost',
            xlabel='epoch',
            ylabel='cost'
        ))
        test_accuracy = vis.line(X=np.array([epoch+1]), Y=np.array([accuracy]),
                                 opts=dict(
            title='Test Accuracy',
            xlabel='epoch',
            ylabel='Accuracy'
        ))
    else:
        vis.updateTrace(X=np.array([epoch+1]), Y=np.array([cost]), win=train_cost)
        vis.updateTrace(X=np.array([epoch+1]), Y=np.array([accuracy]), win=test_accuracy)

# torch.save(cnn.state_dict(), 'cnn.pkl')

