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


def corruptInput(batch_input, corruption_level):
    corruption_matrix = np.random.binomial(1, 1-corruption_level, batch_input.size()[1:])
    corruption_matrix = torch.from_numpy(corruption_matrix).float()
    if torch.cuda.is_available():
        corruption_matrix = corruption_matrix.cuda()
    return batch_input*corruption_matrix


class convVaeNet(nn.Module):
    """
    mostly followed https://github.com/pytorch/examples/blob/master/vae/main.py
    but changed layer to Convolution

    """
    def __init__(self):
        super(convVaeNet, self).__init__()
        self.encode_cnn = nn.Sequential(# ? x 1 x 28 x 28
            nn.Conv2d(1, 5, 5, 1, 0),   # ? x 5 x 24 x 24
            nn.ReLU(),
            nn.MaxPool2d(2),            # ? x 5 x 12 x 12
            nn.Conv2d(5, 10, 3, 1, 0),  # ? x 10 x 10 x 10
            nn.ReLU(),
            nn.MaxPool2d(2),            # ? x 10 x 5 x 5
            nn.Conv2d(10, 20, 3, 1, 0), # ? x 20 x 3 x 3
            nn.ReLU()
        )
        self.encode_fc_mu = nn.Linear(180, 50)
        self.encode_fc_logvar = nn.Linear(180, 50)
        self.decode_fc = nn.Linear(50, 180)
        self.decode_conv_transpose = nn.Sequential( # ? x 20 x 3 x 3
            nn.ConvTranspose2d(20, 10, 3, 2, 1),    # ? x 10 x 5 x 5
            nn.LeakyReLU(),
            nn.ConvTranspose2d(10, 5, 5, 3, 1),     # ? x 5 x 15 x 15
            nn.LeakyReLU(),
            nn.ConvTranspose2d(5, 1, 2, 2, 1),      # ? x 1 x 28 x 28
            nn.Sigmoid()
        )
        # use bilinear/NN sampling for now
        self.decode_conv_bilinear_upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),   # ? x 20 x 6 x 6
            nn.Conv2d(20, 10, 3, 1, 1),                     # ? x 10 x 6 x 6
            nn.LeakyReLU(0.3),
            nn.Upsample(scale_factor=3, mode='bilinear'),   # ? x 10 x 18 x 18
            nn.Conv2d(10, 5, 5, 1, 0),                      # ? x 5 x 14 x 14
            nn.LeakyReLU(0.3),
            nn.Upsample(scale_factor=2, mode='bilinear'),   # ? x 5 x 28 x 28
            nn.Conv2d(5, 1, 3, 1, 1),                       # ? x 1 x 28 x 28
            nn.Sigmoid()
        )
        self.decode_conv = self.decode_conv_transpose

    def encode(self, data):
        x = data
        x = self.encode_cnn(x)
        x = x.view(x.size()[0], -1)
        return self.encode_fc_mu(x), self.encode_fc_logvar(x)

    def decode(self, latent_vector):
        x = latent_vector
        x = self.decode_fc(x)
        x = x.view(x.size()[0], -1, 3, 3)
        x = self.decode_conv(x)
        return x

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, data):
        mu, logvar = self.encode(data)
        latent_vector = self.reparameterize(mu, logvar)
        reconstruction = self.decode(latent_vector)
        reconstruction = reconstruction.view(*data.size())
        return reconstruction, mu, logvar

    def set_decode_conv(self, decode_conv="transpose"):
        if decode_conv == "transpose":
            self.decode_conv = self.decode_conv_transpose
        elif decode_conv == "bilinear-upsample":
            self.decode_conv = self.decode_conv_bilinear_upsample
        elif decode_conv == "perforated-upsample":
            pass # TODO: implement perforated upsample
        elif decode_conv == "nn-upsample":
            pass # TODO: implement nn upsample
        else:
            raise ValueError


MNIST_DSET = os.path.join(DATA_DIR, 'data_mnist')
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


vis = visdom.Visdom()
learning_rate = 1e-3
decay = 1e-5
epochs = 5
corruption_level = 0.4


def Criterion(reconstruct_x, x, mu, logvar):
    cost = F.binary_cross_entropy(reconstruct_x, x)

    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_div /= batch_size * 784

    return cost + kl_div

net = convVaeNet()
net.apply(initializeWeight)
criterion = Criterion
if torch.cuda.is_available():
    net = net.cuda()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate,
                             weight_decay=decay)

# TODO: use logger and timer
def train(epoch, decode_tech="transpose"):
    net.set_decode_conv(decode_tech)
    net.train()
    cost = 0.0
    for batch_idx, (x, _) in enumerate(train_loader):
        if torch.cuda.is_available():
            x = x.cuda()
        X = Variable(corruptInput(x, corruption_level))
        original = Variable(x)
        reconstruction, mu, logvar = net(X)
        loss = criterion(reconstruction, original, mu, logvar)

        net.zero_grad()
        loss.backward()
        optimizer.step()

        cost += loss.data[0]*len(x)
        if batch_idx%100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch + 1, batch_idx * len(x), train_size,
                100. * (batch_idx*len(x)) / train_size, loss.data[0]))
    return cost/train_size

# TODO: print epoch and use logger
def test(epoch):
    net.eval()
    accuracy = 0.0
    for batch_idx, (x, _) in enumerate(test_loader):
        if torch.cuda.is_available():
            x = x.cuda()
        X = Variable(x)
        reconstruction, mu, logvar = net(X)
        loss = criterion(reconstruction, X, mu, logvar)

        accuracy += loss.data[0] * len(x)
    return accuracy / test_size

def show_image(original, title='original-reconstruction', sidebyside=True):
    net.eval()
    x = original
    if torch.cuda.is_available():
        x = x.cuda()
    X = Variable(x)
    reconstruction, _, _ = net(X)
    if sidebyside:
        images = torch.cat([X.data, reconstruction.data], 3).cpu().numpy()
    else:
        images = reconstruction.data.cpu().numpy()
    vis.images(images, nrow=8, opts=dict(title=title))


# get 1 batch for fixed visualization
fixed_image_test, _ = test_loader.__iter__().__next__()
if torch.cuda.is_available(): fixed_image_test = fixed_image_test.cuda()
# show_image(fixed_image_test, 'before-training-original-reconstruction')

print("Starting training")
print(net)
for epoch in range(0, epochs):
    cost = train(epoch, decode_tech="bilinear-upsample")
    accuracy = test(epoch)
    # TODO: generalize and move this to helper with param cost/accuracy, epoch, train/test flag
    if(epoch == 0):
        train_cost = vis.line(X=np.array([epoch+1]), Y=np.array([cost]), opts=dict(
            title='Training Loss',
            xlabel='epoch',
            ylabel='CrossEntropyLoss'
        ))
        test_accuracy = vis.line(X=np.array([epoch+1]), Y=np.array([accuracy]),
                                 opts=dict(
            title='Test Loss',
            xlabel='epoch',
            ylabel='Accuracy'
        ))
    else:
        vis.line(X=np.array([epoch+1]), Y=np.array([cost]), win=train_cost, update='append')
        vis.line(X=np.array([epoch+1]), Y=np.array([accuracy]), win=test_accuracy, update='append')
    if epoch%5==0:
        show_image(fixed_image_test)

show_image(fixed_image_test, "bilinear-reconstruction", False)