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
from definitions import DATA_DIR

vis = visdom.Visdom()

# args
workers = 4
nz = 100    # size of noise latent vector
ngf = 64    # size of generator feature
ndf = 64    # size of discriminator feature
nc = 1      # size of channel
D_lr = 2*1e-4
G_lr = 2*1e-4
beta1 = 0.5
epochs = 25
batch_size = 32

MNIST_DSET = os.path.join(DATA_DIR, 'data_mnist')
train_dataset = datasets.MNIST(MNIST_DSET, train=True,
    transform=transforms.Compose([
        transforms.Scale(ndf),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ]))
train_size = len(train_dataset)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.zero_()

class G(nn.Module):
    def __init__(self):
        super(G, self).__init__()
        self.deconv = nn.Sequential(
            # nz*1
            nn.ConvTranspose2d(nz, ngf*8, 4, 1, 0),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(),
            # nn.LeakyReLU(0.2),
            # (ngf*8, 4, 4)
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(),
            # nn.LeakyReLU(0.2),
            # nn.Dropout2d(),
            # (ngf*4, 8, 8)
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(),
            # nn.LeakyReLU(0.2),
            # (ngf*2, 16, 16)
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1),
            nn.BatchNorm2d(ngf),
            nn.ReLU(),
            # nn.LeakyReLU(0.2),
            # nn.Dropout2d(),
            # (ngf, 32, 32)
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1),
            nn.Tanh()
            # (nc, 64, 64)
        )

    def forward(self, data):
        x = self.deconv(data)
        return x


class D(nn.Module):
    def __init__(self):
        super(D, self).__init__()
        self.conv = nn.Sequential(
            # (nc, 64, 64)
            nn.Conv2d(nc, ndf, 4, 2, 1),
            # nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2),
            # (ndf, 32, 32)
            nn.Conv2d(ndf, ndf*2, 4, 2, 1),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2),
            # (ndf*2, 16, 16)
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2),
            # (ndf*4, 8, 8)
            nn.Conv2d(ndf*4, ndf*8, 4, 2, 1),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2),
            # (ndf*8, 4, 4)
            nn.Conv2d(ndf * 8, 1, 4, 1, 0),
            nn.Sigmoid()
            # (1, 1, 1)
        )

    def forward(self, data):
        x = self.conv(data)
        return x


G_net = G()
D_net = D()
G_net.apply(weights_init)
print(G_net)
D_net.apply(weights_init)
print(D_net)
criterion = nn.BCELoss()

D_optim = torch.optim.Adam(D_net.parameters(), lr=D_lr, betas=(beta1, 0.999))
G_optim = torch.optim.Adam(G_net.parameters(), lr=G_lr, betas=(beta1, 0.999))


# real_label = 1
# fake_label = 0
noise = torch.FloatTensor(batch_size, nz, 1, 1)
fixed_noise = Variable(torch.FloatTensor(batch_size, nz, 1, 1).normal_(0, 1))

if torch.cuda.is_available():
    D_net = D_net.cuda()
    G_net = G_net.cuda()
    criterion = criterion.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

iteration = 0
for epoch in range(epochs):
    for batch_idx, train_batch in enumerate(train_loader):
        # update Discriminator
        D_net.zero_grad()
        # real data
        data, _ = train_batch
        current_batch_size = data.size(0)
        real_label = torch.ones(current_batch_size)
        fake_label = torch.zeros(current_batch_size)

        real_label, fake_label = Variable(real_label), Variable(fake_label)
        input_var = Variable(data)
        if torch.cuda.is_available():
            real_label, fake_label, input_var = real_label.cuda(), fake_label.cuda(), input_var.cuda()

        output = D_net(input_var).squeeze()
        costD_real = criterion(output, real_label)
        D_x = output.data.mean()

        # fake data
        noise.resize_(current_batch_size, nz, 1, 1).normal_(0, 1)
        noise_var = Variable(noise)
        
        fake_input_var = G_net(noise_var)
        output = D_net(fake_input_var).squeeze()
        costD_fake = criterion(output, fake_label)
        D_G_z1 = output.data.mean()

        costD = costD_real + costD_fake
        costD.backward()
        D_optim.step()

        # update Generator
        G_net.zero_grad()
        noise.resize_(current_batch_size, nz, 1, 1).normal_(0, 1)
        noise_var = Variable(noise)

        fake_input_var = G_net(noise_var)
        output = D_net(fake_input_var).squeeze()
        costG = criterion(output, real_label)
        costG.backward()
        D_G_z2 = output.data.mean()

        G_optim.step()

        print("[{}/{}][{}/{}] Loss_D: {:.4f} Loss_G: {:.4f} D(x): {:.4f} D(G(z)): {:.4f} / {:.4f}"
            .format(epoch+1, epochs, batch_idx, len(train_loader), costD.data[0], costG.data[0], D_x, D_G_z1, D_G_z2))

        cost = np.array([np.concatenate((costD.data.cpu().numpy(), costG.data.cpu().numpy()))])
        # print(cost)
        if(iteration == 0):
            vis.images(input_var.data.cpu().numpy(), nrow=4, opts=dict(caption="real sample"))
            win = vis.line(X=np.array([[0,0]]), Y=cost, opts=dict(
                xlabel='iteration',
                ylabel='cost',
                legend=['costD', 'costG']
            ))
        else:
            vis.line(X=np.array([[iteration, iteration]]), Y=cost, win=win, update="append")

        if((batch_idx % 625) == 0):
            output = G_net(fixed_noise)
            vis.images(output.data.cpu().numpy(), nrow=4, opts=dict(caption="{}.{}".format(epoch+1, batch_idx)))

        iteration += 1
