# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import matplotlib.pyplot as plt
import time
import visdom
import numpy as np

from torchvision import datasets, transforms
from torch.autograd import Variable


train_dataset = datasets.MNIST('./data_mnist/', train=True, transform=transforms.ToTensor())
test_dataset = datasets.MNIST('./data_mnist/', train=False, transform=transforms.ToTensor())
train_size = len(train_dataset)
test_size = len(test_dataset)

batch_size = 16
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)


class convNet(nn.Module):
    def __init__(self):
        super(convNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),
            nn.MaxPool2d(2),
            nn.LeakyReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 12, kernel_size=5),
            nn.MaxPool2d(2),
            nn.LeakyReLU()
        )
        self.fc1 = nn.Linear(4*4*12, 50)
        self.fc2 = nn.Linear(50, 10)
        
    def forward(self, data):    # (?, 28, 28, 1)
        x = self.conv1(data)    # (?, 24, 24, 6) > (?, 12, 12, 6)
        x = self.conv2(x)       # (?, 8, 8, 12) > (?, 4, 4, 12)
        x = x.view(-1, 192)     # (?, 192)
        x = F.dropout(self.fc1(x), training=self.training)  # (?, 50)
        x = F.leaky_relu(x)     # (?, 50)
        x = self.fc2(x)         # (?, 10)
        return x

vis = visdom.Visdom()

dtype = torch.FloatTensor
epochs = 400
learning_rate = 1e-3

cnn = convNet()
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss()


def train(epoch):
    epoch_start = time.time()
    cnn.train()
    cost = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()

        pred = cnn(data)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 500 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch+1, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))
        cost += loss.data[0]*len(data)
    print("time for training epoch: {}".format(time.time()-epoch_start))
    return cost/train_size


def eval(epoch):
    cnn.eval()
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data), target
        output = cnn(data)
        _, pred = torch.max(output.data, 1)
        correct += (pred == target).sum()
    print("Test accuracy for epoch {} : {}".format(epoch+1, 100*correct/test_size))
    return 100*correct/test_size


for epoch in range(0, epochs):
    cost = train(epoch)
    accuracy = eval(epoch)
    if(epoch == 0):
        train_cost = vis.line(X=np.array([epoch+1]), Y=np.array([cost]), opts=dict(
            title='Train Cost',
            xlabel='epoch',
            ylabel='cost'
        ))
        test_accuracy = vis.line(X=np.array([epoch+1]), Y=np.array([accuracy]), opts=dict(
            title='Test Accuracy',
            xlabel='epoch',
            ylabel='Accuracy'
        ))
    else:
        vis.updateTrace(X=np.array([epoch+1]), Y=np.array([cost]), win=train_cost)
        vis.updateTrace(X=np.array([epoch+1]), Y=np.array([accuracy]), win=test_accuracy)

torch.save(cnn.state_dict(), 'cnn.pkl')