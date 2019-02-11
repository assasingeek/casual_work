#This is implementation of XOR logic using Radial Basis Function Network (RBFN).
# This gives approx. 98% accuracy.

import logging as lg
lg.basicConfig(level=lg.INFO, format='%(levelname)-8s: %(message)s')
from sklearn.cluster import KMeans
import numpy as np
import torch

class XOR(torch.nn.Module):
    def __init__(self, num_dims=2, sigma=1.3):
        super().__init__()
        self.sigma = sigma
        mu = (torch.rand(1, 2) > 0.5).float()
        sig = 0.075
        # self.c = torch.nn.Parameter(mu + sig * torch.randn(2, 2))
        # self.w = torch.nn.Parameter(sig * torch.randn(2, 1))
        self.lr1 = (0.3)
        self.lr2 = (0.0002)
        # self.c = (torch.Tensor.float(torch.randint(2, (50,2))))
        # self.c = (torch.Tensor.float(mu + sig * torch.randn(1, 2)))
        self.w = (sig * torch.randn(2, 500))
        self.test_l = torch.nn.Linear(1, 1)
        self.flag = 0
        
    def forward(self, x, y) :
        self.c = self.k_cluster(x)
        x = self.RBF(x, y)
        x = self.test_l(x)
        return x

    def k_cluster(self, x):

        if(self.flag == 1):
            return self.c
        self.flag = 1
        kmeans = KMeans(n_clusters=4).fit(x)
        self.c = kmeans.cluster_centers_
        self.c = torch.Tensor(self.c)
        print('CLUSTER')
        print(self.c)
        exit(0)
        return self.c


    def RBF(self, x, y):

        phi = torch.exp(-((x-self.c)**2))/(self.sigma ** 2)
        phi = torch.nn.functional.normalize(phi, p=1 , dim=0)
        y_hat = torch.mm(phi, self.w)
        # print(y_hat.shape)

        y_hat = y_hat[0]
        # y_hat.reshape(500,1)
        a = (self.lr1 * ( phi * (x-self.c)))
        # print(y_hat)
        # exit(0)
        self.c = self.c + ((y-y_hat).transpose(0,1) * (torch.mm(a.transpose(0,1), self.w.transpose(0,1))))/(2 * (self.sigma ** 2))
        # print(self.c)

        # print((y-y_hat).shape)
        # print(phi.shape)
        # exit(0)
        self.w = self.w + (self.lr2 * torch.mm(phi.transpose(0, 1), (y-y_hat)))

        return y_hat


def data(sizes=100) :
    # x = torch.randint(2, (sizes,1), requires_grad=True)
    mu = (torch.rand(sizes, 2) > 0.5).float()
    sigma = 0.075
    x = mu + sigma * torch.randn(sizes, 2)
    y = (torch.abs(1 - mu.sum(1)) < 1).float().reshape(50000,1)
    i=0
    return x,y

if __name__=='__main__' :
    model = XOR()
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99, eps=1e-08)
    nb_epochs = 100
    data_size = 50000
    batch_size = int(data_size/nb_epochs)
    for epoch in range(nb_epochs):
        X, Y = data(data_size)

        for i, i0 in enumerate(range(0, data_size, batch_size)) :
            i1 = i0 + batch_size
            x, y = X[i0:i1], Y[i0:i1]

            optimizer.zero_grad()
            # print(x.shape)
            y_hat = model(torch.Tensor.float(x), y)

            loss = loss_fn(y_hat, y)
            # accuracy = (torch.abs(quant - y) < 1e-2).float().mean()
            # accuracy = ((y_hat - y) < 1.5e-1).float().mean()
            y_hat = y_hat
            accuracy = (torch.abs(torch.abs(y_hat) - torch.abs(y)) < 1.5e-1).float().mean()
            # accuracy = (torch.abs(y_hat - y) < 4.5e-1).float().mean()

            # loss.backward()
            # report batch stats
            # lg.info(f'Iter:{i} loss:{loss:%08.4f}') #' accuracy:{100*accuracy:%5.2f}')
            lg.info('[%02d:%03d] loss:%8.4f accuracy:%5.2f',
                epoch, i, loss.item(), 100*accuracy.item())

            # update
            optimizer.step()

        # print("Epoch: {} Loss: {}".format(epoch, loss))

