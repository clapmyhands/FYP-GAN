# coding: utf-8
import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import visdom
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable

sys.path.append(os.path.join(os.path.dirname(__file__), "..",".."))
from util.helper import initializeXavierUniformWeight
from definitions import DATA_DIR

from util.grid_graph import grid_graph
from util.coarsening import coarsen, lmax_L, perm_data, rescale_L, unperm_indices, unperm_data

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


# TODO: move graph dataset to util
class MnistGraphDataset(Dataset):
    def __init__(self, data, labels, train=True, transform=None):
        self.train = train
        self.transform = transform

        if self.train:
            self.train_data, self.train_labels = data, labels
        else:
            self.test_data, self.test_labels = data, labels

    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]


        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)


MNIST_DSET = os.path.join(DATA_DIR, "data_mnist")
train_dataset = datasets.MNIST(MNIST_DSET, train=True,download=True)
test_dataset = datasets.MNIST(MNIST_DSET, train=False,download=True)
train_size = len(train_dataset)
test_size = len(test_dataset)
# Move mnist to new graph Dataset
data = train_dataset.train_data.view(train_size, -1).type(dtypeFloat)
data = data/255
labels = train_dataset.train_labels
train_dataset = MnistGraphDataset(data,labels,train=True)

data = test_dataset.test_data.view(test_size, -1).type(dtypeFloat)
data = data/255
labels = test_dataset.test_labels
test_dataset = MnistGraphDataset(data,labels,train=False)

################ Construct graph #################
# TODO: move graph construction to util
t_start = time.time()
grid_side = 28
number_edges = 8
metric = 'euclidean'
A = grid_graph(grid_side,number_edges,metric) # create graph of Euclidean grid

# Compute coarsened graphs
coarsening_levels = 4
L, perm = coarsen(A, coarsening_levels)
unperm = unperm_indices(perm)

# Compute max eigenvalue of graph Laplacians
lmax = []
for i in range(coarsening_levels):
    lmax.append(lmax_L(L[i]))
print('lmax: ' + str([lmax[i] for i in range(coarsening_levels)]))

# Reindex nodes to satisfy a binary tree structure
train_dataset.train_data = perm_data(train_dataset.train_data, perm)
test_dataset.test_data = perm_data(test_dataset.test_data, perm)

print('Execution time: {:.2f}s'.format(time.time() - t_start))

################ learning parameters #################
learning_rate = 0.05
dropout_value = 0.5
corruption_level = 0.4
l2_decay = 5e-4
batch_size = 100
epochs = 20
nb_iter = int(epochs * train_size) // batch_size
print('num_epochs=', epochs, ', train_size=', train_size, ', nb_iter=', nb_iter)

print("train dataset size: {}".format(train_dataset.train_data.shape))
print("test dataset size: {}".format(test_dataset.test_data.shape))

train_loader = DataLoader(train_dataset, batch_size=batch_size,
                               shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=4)


class sparse_mm(torch.autograd.Function):
    """
    Implementation of a new autograd function for sparse variables,
    called "my_sparse_mm", by subclassing torch.autograd.Function
    and implementing the forward and backward passes.
    """

    def forward(self, W, x):  # W is SPARSE
        self.save_for_backward(W, x)
        y = torch.mm(W, x)
        return y

    def backward(self, grad_output):
        W, x = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input_dL_dW = torch.mm(grad_input, x.t())
        grad_input_dL_dx = torch.mm(W.t(), grad_input)
        return grad_input_dL_dW, grad_input_dL_dx


def corruptInput(batch_input, corruption_level):
    corruption_matrix = np.random.binomial(1, 1-corruption_level, batch_input.size()[1:])
    corruption_matrix = torch.from_numpy(corruption_matrix).float()
    if torch.cuda.is_available():
        corruption_matrix = corruption_matrix.cuda()
    return batch_input*corruption_matrix


class graph_convVaeNet(nn.Module):
    def __init__(self, net_parameters):
        super(graph_convVaeNet, self).__init__()
        # parameters
        D, CL1_F, CL1_K, CL2_F, CL2_K, FC1_F, FC2_F = net_parameters
        FC1_Fin = CL2_F * (D // 16)

        self.CL1_K = CL1_K
        self.CL1_F = CL1_F
        self.CL2_K = CL2_K
        self.CL2_F = CL2_F
        self.FC1Fin = FC1_Fin

        self.cl1 = nn.Linear(CL1_K, CL1_F)   # B x V x CL1_F
        self.cl2 = nn.Linear(CL2_K * CL1_F, CL2_F)  # B x V/4 x CL2_F
        self.encode_fc1 = nn.Linear(FC1_Fin, FC1_F)  # B x FC1_F
        self.encode_fc2_mu = nn.Linear(FC1_F, FC2_F)  # B x FC2_F
        self.encode_fc2_logvar = nn.Linear(FC1_F, FC2_F)  # B x FC2_F

        self.decode_fc1 = nn.Linear(FC2_F, FC1_F)  # B x FC1_F
        self.decode_fc2 = nn.Linear(FC1_F, FC1_Fin)  # B x FC1_Fin
        # B x V/16*CL2_F > B x V/16 x CL2_F > B x V/4 x CL2_F
        self.decode_cl1 = nn.Linear(CL2_K * CL2_F, CL1_F) # B x V/4 x CL1_F
        self.decode_cl2 = nn.Linear(CL1_K * CL1_F, 1) # B x V x 1

        nb_param = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("nb of param: {}".format(nb_param))

        self.BCELoss = nn.BCELoss()

    def graph_conv_cheby(self, x, cl, L, lmax, Fout, K):
        # parameters
        # B = batch size
        # V = nb vertices
        # Fin = nb input features
        # Fout = nb output features
        # K = Chebyshev order & support size
        B, V, Fin = x.size()
        B, V, Fin = int(B), int(V), int(Fin)

        # rescale Laplacian
        lmax = lmax_L(L)
        L = rescale_L(L, lmax)

        # convert scipy sparse matric L to pytorch
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col)).T
        indices = indices.astype(np.int64)
        indices = torch.from_numpy(indices)
        indices = indices.type(torch.LongTensor)
        L_data = L.data.astype(np.float32)
        L_data = torch.from_numpy(L_data)
        L_data = L_data.type(torch.FloatTensor)
        L = torch.sparse.FloatTensor(indices, L_data, torch.Size(L.shape))
        L = Variable(L, requires_grad=False)
        if torch.cuda.is_available():
            L = L.cuda()

        # transform to Chebyshev basis
        x0 = x.permute(1, 2, 0).contiguous()  # V x Fin x B
        x0 = x0.view([V, Fin * B])  # V x Fin*B
        x = x0.unsqueeze(0)  # 1 x V x Fin*B

        if K > 1:
            x1 = sparse_mm()(L, x0)  # V x Fin*B
            x = torch.cat((x, x1.unsqueeze(0)), 0)  # 2 x V x Fin*B
        for k in range(2, K):
            x2 = 2 * sparse_mm()(L, x1) - x0
            x = torch.cat((x, x2.unsqueeze(0)), 0)  # M x Fin*B
            x0, x1 = x1, x2

        x = x.view([K, V, Fin, B])  # K x V x Fin x B
        x = x.permute(3, 1, 2, 0).contiguous()  # B x V x Fin x K
        x = x.view([B * V, Fin * K])  # B*V x Fin*K

        # Compose linearly Fin features to get Fout features
        x = cl(x)  # B*V x Fout
        x = x.view([B, V, Fout])  # B x V x Fout

        return x

    def graph_max_pool(self, x, p):
        if p > 1:
            x = x.permute(0, 2, 1).contiguous()  # x = B x F x V
            x = nn.MaxPool1d(p)(x)  # B x F x V/p
            x = x.permute(0, 2, 1).contiguous()  # x = B x V/p x F
            return x
        else:
            return x

    def graph_unpool(self, x, p):
        if p > 1:
            x = x.permute(0, 2, 1).contiguous()  # x = B x F x V
            B, F, V = x.size()
            x = x.view([B,F,V,1]).repeat(1, 1, 1, p)
            x = x.view([B,F,-1]) # B x F x V*p
            x = x.permute(0, 2, 1).contiguous()  # x = B x V*p x F
            return x
        else:
            return x

    def encode(self, data, d, L, lmax):
        # graph CL1
        x = data.unsqueeze(2)  # B x V x Fin=1
        x = self.graph_conv_cheby(x, self.cl1, L[0], lmax[0], self.CL1_F, self.CL1_K)
        x = self.graph_max_pool(F.relu(x), 4)  # B x V/4 x CL1_F
        # graph CL2
        x = self.graph_conv_cheby(x, self.cl2, L[2], lmax[2], self.CL2_F, self.CL2_K)
        x = self.graph_max_pool(F.relu(x), 4)  # B x V/16 x CL2_F
        # FC1
        x = x.view(-1, self.FC1Fin)
        x = F.relu(self.encode_fc1(x))
        # x = nn.Dropout(d)(x)

        return self.encode_fc2_mu(x), self.encode_fc2_logvar(x)

    def decode(self, latent_vector, d, L, lmax):
        z = F.relu(self.decode_fc1(latent_vector))
        z = self.decode_fc2(z)
        z = z.view(-1, D//16, self.CL2_F)  # B x V/16 x CL2_F
        z = self.graph_unpool(z, 4)  # B x V/4 x CL2_F
        z = self.graph_conv_cheby(z, self.decode_cl1, L[2], lmax[2], self.CL1_F, self.CL2_K)
        z = F.relu(z)
        z = self.graph_unpool(z, 4)  # B x V x CL1_F
        z = self.graph_conv_cheby(z, self.decode_cl2, L[0], lmax[0], 1, self.CL1_K)
        z = F.sigmoid(z)
        z = z.squeeze(2)
        return z

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, data, d, L, lmax):
        mu, logvar = self.encode(data, d, L, lmax)
        latent_vector = self.reparameterize(mu, logvar)
        reconstruction = self.decode(latent_vector, d, L, lmax)
        return reconstruction, mu, logvar

    def loss(self, y, y_target, mu, logvar, l2_decay=1e-5):
        loss = self.BCELoss(y, y_target)
        # loss = F.binary_cross_entropy(y, y_target, size_average=False)

        l2_loss = 0.0
        for param in self.parameters():
            data = 0.5 * l2_decay * param * param
            l2_loss += data.sum()
        loss += l2_loss

        # kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # loss += kl_div

        return loss

    def update(self, lr, decay):
        update = torch.optim.Adam(self.parameters(), lr=lr)
        return update

    def update_learning_rate(self, optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return optimizer


vis = visdom.Visdom()

D = train_dataset.train_data.shape[1]
CL1_F = 25
CL1_K = 25
CL2_F = 50
CL2_K = 25
FC1_F = 400
FC2_F = 64
net_parameters = [D, CL1_F, CL1_K, CL2_F, CL2_K, FC1_F, FC2_F]

net = graph_convVaeNet(net_parameters)
net.apply(initializeXavierUniformWeight)

# Optimizer
global_lr = learning_rate
decay = 0.95
lr = learning_rate
if torch.cuda.is_available():
    net = net.cuda()
optimizer = net.update(lr, l2_decay)

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

def show_image(graph_data, d, L, lmax, unperm, title='original', reconstruction=True):
    net.eval()
    x = Variable(graph_data.type(dtypeFloat))
    if reconstruction:
        reconstruct_img, _, _ = net.forward(x, d, L, lmax)
    else:
        reconstruct_img = x
    # images = perm_data(reconstruct_img.data.cpu(), unperm)[:,:784]
    images = unperm_data(reconstruct_img.data.cpu(), unperm, 784)
    images = images.reshape((graph_data.shape[0], 1, 28, 28)) # reshape to picture
    vis.images(images, nrow=10, opts=dict(title=title))

print("Starting training")
print(net)

fixed_image_test, _ = test_loader.__iter__().__next__()
show_image(fixed_image_test, dropout_value, L, lmax, unperm, reconstruction=False)
show_image(fixed_image_test, dropout_value, L, lmax, unperm, "reconstruction")
for epoch in range(epochs):
    epoch_start = time.time()
    net.train()
    cost = 0.0
    for batch_idx, (x, _) in enumerate(train_loader):
        x = x.type(dtypeFloat)
        X = Variable(corruptInput(x, corruption_level))
        original = Variable(x)

        reconstruction, mu, logvar = net.forward(X, dropout_value, L, lmax)
        loss = net.loss(reconstruction, original, mu, logvar, l2_decay)
        # loss = criterion(reconstruction, original, mu, logvar)
        train_loss = loss.data[0]
        cost += train_loss

        net.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{:5}/{:5} ({:2.0f}%)]\tLoss: {:10.6f}'.format(
                epoch + 1, batch_idx * len(x), train_size,
                100. * batch_idx / len(train_loader), train_loss/len(x)))

    t_stop = time.time() - epoch_start
    print('epoch= %d, loss(train)= %.6f, time= %.3f, lr= %.5f' %
          (epoch + 1, cost / train_size, t_stop, lr))

    # update learning rate
    lr = global_lr * pow(decay, epoch + 1)
    optimizer = net.update_learning_rate(optimizer, lr)

    show_image(fixed_image_test, dropout_value, L, lmax, unperm, "reconstruction")

