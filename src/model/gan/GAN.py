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
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), "..",".."))
# from util.helper import initializeXavierUniformWeight
from definitions import DATA_DIR

if torch.cuda.is_available():
    print('cuda available')
    dtypeFloat = torch.cuda.FloatTensor
    dtypeLong = torch.cuda.LongTensor
    torch.cuda.manual_seed(1)
else:
    print('cuda not available')
    dtypeFloat = torch.FloatTensor
    dtypeLong = torch.LongTensor
    torch.manual_seed(1)

vis = visdom.Visdom()

# args
workers = 4
nz = 64    # size of noise latent vector
ngf = 64    # size of generator feature
ndf = 64    # size of discriminator feature
nc = 1      # size of channel
D_lr = 2e-4
G_lr = 2e-4
beta1 = 0.5
epochs = 25
batch_size = 64

MNIST_DSET = os.path.join(DATA_DIR, 'data_mnist')
train_dataset = datasets.MNIST(
    MNIST_DSET,
    train=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]))
train_size = len(train_dataset)
train_loader = DataLoader(train_dataset, batch_size=batch_size,
                          shuffle=True, num_workers=workers)

def rescaleImage(x):
    x = (x*0.5) + 0.5
    return x

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0, 0.02)
        m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        m.weight.data.normal_(0, 0.02)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.02)
        m.bias.data.zero_()


class G(nn.Module):
    def __init__(self):
        super(G, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1),
            nn.BatchNorm2d(ngf),
            nn.ReLU(),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1),
            nn.Tanh()
        )
        self.fc = nn.Sequential(
            nn.Linear(nz, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, ngf*2 * 7 * 7),
            nn.BatchNorm1d(ngf*2 * 7 * 7),
            nn.ReLU(),
        )

    def forward(self, data):
        x = self.fc(data)
        x = x.view(-1, ngf*2, 7, 7)
        x = self.deconv(x)
        return x


class D(nn.Module):
    def __init__(self):
        super(D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(ndf, ndf*2, 4, 2, 1),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Sequential(
            nn.Linear(ndf*2*7*7, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, data):
        x = self.conv(data)
        x = x.view(-1, ndf*2*7*7)
        x = self.fc(x)
        return x


G_net = G()
G_net.apply(weights_init)
print(G_net)
D_net = D()
D_net.apply(weights_init)
print(D_net)
criterion = nn.BCELoss()

if torch.cuda.is_available():
    D_net = D_net.cuda()
    G_net = G_net.cuda()
    criterion = criterion.cuda()

D_optim = torch.optim.Adam(D_net.parameters(), lr=D_lr, betas=(beta1, 0.999))
G_optim = torch.optim.Adam(G_net.parameters(), lr=G_lr, betas=(beta1, 0.999))

# real_label = 1
# fake_label = 0
noise = torch.FloatTensor(batch_size, nz).type(dtypeFloat)
fixed_noise = Variable(torch.FloatTensor(batch_size, nz)
                       .normal_(0, 1).type(dtypeFloat))

iteration = 0
for epoch in range(epochs):
    for batch_idx, train_batch in enumerate(train_loader):
        # real data
        data, _ = train_batch
        current_batch_size = data.size(0)
        real_label = torch.ones(current_batch_size).type(dtypeFloat)
        fake_label = torch.zeros(current_batch_size).type(dtypeFloat)

        real_label, fake_label = Variable(real_label), Variable(fake_label)
        input_var = Variable(data.type(dtypeFloat))

        output = D_net(input_var).squeeze()
        costD_real = criterion(output, real_label)
        D_x = output.data.mean()

        # fake data
        noise.resize_(current_batch_size, nz).normal_(0, 1)
        noise_var = Variable(noise.type(dtypeFloat))
        
        fake_input_var = G_net(noise_var)
        output = D_net(fake_input_var).squeeze()
        costD_fake = criterion(output, fake_label)
        D_G_z1 = output.data.mean()

        # update Discriminator
        costD = costD_real + costD_fake
        D_net.zero_grad()
        costD.backward()
        D_optim.step()

        noise.resize_(current_batch_size, nz).normal_(0, 1)
        noise_var = Variable(noise.type(dtypeFloat))

        fake_input_var = G_net(noise_var)
        output = D_net(fake_input_var).squeeze()
        costG = criterion(output, real_label)
        D_G_z2 = output.data.mean()

        # update Generator
        G_net.zero_grad()
        costG.backward()
        G_optim.step()

        print("[{}/{}][{}/{}] Loss_D: {:.4f} Loss_G: {:.4f} D(x): {:.4f} D(G(z)): {:.4f} / {:.4f}"
            .format(epoch+1, epochs, batch_idx, len(train_loader), costD.data[0], costG.data[0], D_x, D_G_z1, D_G_z2))

        cost = np.array([np.concatenate((costD.data.cpu().numpy(), costG.data.cpu().numpy()))])
        if(iteration == 0):
            # images = rescaleImage(input_var.data.cpu().numpy())
            # vis.images(images, nrow=4, opts=dict(caption="real sample"))
            win = vis.line(X=np.array([[0,0]]), Y=cost, opts=dict(
                xlabel='iteration',
                ylabel='cost',
                legend=['costD', 'costG']
            ))
        else:
            vis.line(X=np.array([[iteration, iteration]]), Y=cost, win=win, update="append")

        if((batch_idx % 625) == 0):
            output = G_net(fixed_noise)
            images = rescaleImage(output.data.cpu().numpy())
            vis.images(images, nrow=4, opts=dict(caption="{}.{}".format(epoch+1, batch_idx)))

        iteration += 1