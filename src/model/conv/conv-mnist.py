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
from util.helper import initializeXavierNormalWeight
from definitions import DATA_DIR

def normalizeForImage(tensor: torch.FloatTensor):
    batch_size = tensor.size()[0]
    max_val, _ = torch.max(tensor.view(batch_size, -1), 1, keepdim=True)
    min_val, _ = torch.min(tensor.view(batch_size, -1), 1, keepdim=True)
    return (tensor-min_val)/(max_val-min_val+1e-12), max_val, min_val

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
        self.convo1 = nn.Conv2d(1, 10, kernel_size=5)
        self.convo2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(20*4*4, 100)
        self.fc2 = nn.Linear(100, 10)

    def conv_layer(self, data, layer=1):
        convo = None
        if layer==1:
            convo = self.convo1
        elif layer==2:
            convo = self.convo2
        x = data
        x = convo(x)
        x = F.dropout2d(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        return x

    def forward(self, data):            # (?, 1, 28, 28)
        x = self.conv_layer(data, 1)    # (?, 10, 24, 24) > (?, 10, 12, 12)
        x = self.conv_layer(x, 2)       # (?, 10, 8, 8) > (?, 20, 4, 4)
        x = x.view(-1, 320)             # (?, 320)
        x = self.fc1(x)                 # (?, 100)
        x = F.dropout(x, training=self.training)
        x = F.relu(x)                   # (?, 100)
        x = self.fc2(x)                 # (?, 10)
        return x

    def get_conv_activation(self, data):
        x = data
        h1 = F.relu(self.convo1(x))
        x = F.max_pool2d(h1, 2)
        h2 = F.relu(self.convo2(x))
        return h1, h2


vis = visdom.Visdom()

# TODO: try out if dtype can be used instead of .cuda()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

epochs = 100
learning_rate = 1e-3

net = convNet()
net.apply(initializeXavierNormalWeight)
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
        vis.line(X=np.array([epoch+1]), Y=np.array([cost]), win=train_cost, update='append')
        vis.line(X=np.array([epoch+1]), Y=np.array([accuracy]), win=test_accuracy, update='append')

for x, _ in test_loader:
    x = x[0:1] # get 1 example data
    x = x.cuda() if torch.cuda.is_available() else x
    X = Variable(x)
    h1, h2 = net.get_conv_activation(X)
    h1 = normalizeForImage(h1.cpu().data)[0].view(10, 1, 24, 24).numpy()
    h2 = normalizeForImage(h2.cpu().data)[0].view(20, 1, 8, 8).numpy()
    h1 = np.kron(h1, np.ones((1, 1, 4, 4))) # upsize by 4
    h2 = np.kron(h2, np.ones((1, 1, 4, 4))) # upsize by 4
    vis.images(h1, 5)
    vis.images(h2, 5)
    break
