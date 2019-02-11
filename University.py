import logging as lg
lg.basicConfig(level=lg.INFO, format='%(levelname)-8s: %(message)s')
import numpy as np
import torch
from argparse import Namespace
import matplotlib.pyplot as plt
import pandas as pd
# import np_utils


class AdmNN(torch.nn.Module) :
  def __init__(self, num_dims=6, batch_size=20) :
    super().__init__()
    self.sig = 0.7
    self.num_dims = num_dims
    self.w = torch.nn.Parameter(torch.randn(num_dims, 1)) # standard normal distribution : mean=0, variance=1
    self.b = torch.nn.Parameter(self.sig * torch.randn(batch_size, 1))
    self.l1 = torch.nn.Linear(num_dims, batch_size, bias=True)
    # self.l2 = torch.nn.Linear(batch_size, 4, bias=True)
    # self.l3 = torch.nn.Softmax()

  def forward(self,x) :
    # x = self.l1(x)
    # x = self.l2(x)
    # print(x)
    # print(x.shape)
    # x = torch.sum(x,1)
    # print(x)
    # print(x.shape)
    # exit(0)
    x = self.custom(x)
    # x = self.l3(x)
    return x


  def custom(self,x) :
    y = torch.mm(x,self.w) + self.b
    return y/100



if __name__ == '__main__' :

  opt = Namespace()

  data = pd.read_csv('Admission_Data_Training.csv') 
  data.drop(columns=['Serial No.'], axis=1, inplace=True)
  data.drop(columns=['Research'], axis=1, inplace=True)
  X = data.to_numpy()
  s = data[data["Chance of Admit"] >= 0.75]["University Rating"].value_counts().head(5)
  plt.title("University Ratings of Candidates with an 75% acceptance chance")
  s.plot(kind='bar',figsize=(20, 10))
  plt.xlabel("University Rating")
  plt.ylabel("Candidates")
  plt.show()

  opt.num_charac = int(len(X[0])-1)
  opt.num_epochs = 35
  opt.batch_size = 2
  opt.num_samples = len(X)
  opt.num_dims = opt.num_charac
  opt.lr_sgd = 0.01
  opt.lr_adam = 0.4
  opt.momentum = 0.9

  Y = X[:,opt.num_charac]
  data.drop(columns=['Chance of Admit'], axis=7, inplace=True)
  X = data.to_numpy()
  np.resize(X,(len(X),8))
  X = torch.Tensor(X)
  Y = torch.Tensor(Y)
  X = X / X.max(0, keepdim=True)[0]

  f = AdmNN(opt.num_dims, opt.batch_size)
  loss_fn = torch.nn.MSELoss()

  # optimizer = torch.optim.SGD(f.parameters(), lr=opt.lr_sgd,weight_decay=1e-02, momentum=opt.momentum, nesterov=True)
  optimizer = torch.optim.Adam(f.parameters(), lr=opt.lr_adam, betas=(0.7, 0.799), eps=1e-06)
  # optimizer = torch.optim.Adam(f.parameters(), lr=opt.lr_adam, betas=(0.7, 0.799), eps=1e-09)
  losses = []
  for epoch in range(opt.num_epochs) :
    loss_collection = []
    accuracy_collection = []
    for i, i0 in enumerate(range(0, opt.num_samples, opt.batch_size)) :
      i1 = i0 + opt.batch_size
      x, y = X[i0:i1], Y[i0:i1]
      # torch.reshape(y,(20,1))
      # y = torch.Tensor(y)
      # print(y)
      # exit(0)

      optimizer.zero_grad()
      # print(y.shape)
      y_hat = f(x)
      # print(y_hat)


      # y_hat = torch.Tensor.norm(y_hat, dim=1)
      y_hat = y_hat / y_hat.max(0, keepdim=True)[0]
      y = y / y.max(0, keepdim=True)[0]
      # print(y_hat)
      # exit(0)

      loss = loss_fn(y_hat, y)
      accuracy = (torch.abs((y_hat) - y) < 5e-2).float().mean()

      loss.backward()

      # update
      optimizer.step()
      loss_collection.append(loss.item())
      accuracy_collection.append(accuracy.item())
    # report epoch stats
    lg.info('Epoch:%02d loss:%8.4f accuracy:%5.2f',
            epoch, torch.tensor(loss_collection).mean().item(),
            100 * torch.tensor(accuracy_collection).mean().item())
    losses.append(loss_collection)

  torch.save(f.state_dict(), 'Univer.pt')

  f.eval()


  # lo = np.reshape(losses, (-1,))
  # lo = lo[3500-35:3500]
  # i = np.arange(0,opt.num_epochs*(opt.num_samples/opt.batch_size)/100)
  # # lo = lo
  # plt.plot(list(i), lo)
  # plt.ylabel('Losses')
  # plt.xlabel('No of Data')
  # plt.show()
  