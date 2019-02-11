import logging as lg
lg.basicConfig(level=lg.INFO, format='%(levelname)-8s: %(message)s')
import numpy as np
import torch
from argparse import Namespace
import matplotlib.pyplot as plt

class Paraboloid(torch.nn.Module) :
  def __init__(self, num_dims=2) :
    super().__init__()
    self.num_dims = num_dims
    self.a = torch.nn.Parameter(torch.randn(num_dims)) # standard normal distribution : mean=0, variance=1
    self.b = torch.nn.Parameter(torch.full((num_dims,), -0.05))

  def forward(self, x) :
    # print(self.a)
    # print(self.b)
    return (self.a * x + self.b) ** 2

def dataset(a, b, num_samples=100, num_dims=2, scale=100) :
  x = torch.rand(num_samples, num_dims)
  x = scale * torch.rand_like(x) * (2 * x - 1)
  y = (a*x + b) ** 2
  return x, y

if __name__ == '__main__' :
  opt = Namespace()
  opt.num_epochs = 8
  opt.batch_size = 32
  opt.num_samples = 2560
  opt.num_dims = 2
  opt.data_scale = 1/100
  opt.lr_sgd = 0.999
  opt.momentum = 0.9
  lg.info(f'{opt}')

  opt.a, opt.b = torch.rand(2,2)/100
  lg.info(f'a:{opt.a} b:{opt.b}')
  X, Y = dataset(opt.a, opt.b, opt.num_samples, 
                 opt.num_dims, opt.data_scale)
  f = Paraboloid(opt.num_dims)
  loss_fn = torch.nn.MSELoss()
  optimizer = torch.optim.SGD(f.parameters(), lr=opt.lr_sgd, momentum=opt.momentum)

  # losses = []
  for epoch in range(opt.num_epochs) :
    loss_collection = []
    accuracy_collection = []
    for i, i0 in enumerate(range(0, opt.num_samples, opt.batch_size)) :
      i1 = i0 + opt.batch_size
      x, y = X[i0:i1], Y[i0:i1]
      
      optimizer.zero_grad()
      y_hat = f(x)
      # f.train()
      loss = loss_fn(y_hat, y)
      accuracy = (torch.abs(y_hat - y) < 1e-4).float().mean()

      # report batch stats
      # lg.info(f'Iter:{i} loss:{loss:%08.4f}') #' accuracy:{100*accuracy:%5.2f}')
      lg.info('[%02d:%03d] loss:%8.4f accuracy:%5.2f',
              epoch, i, loss.item(), 100*accuracy.item())

      loss.backward()
      # update
      optimizer.step()
      loss_collection.append(loss.item())
      accuracy_collection.append(accuracy.item())

    lg.info('Epoch:%02d loss:%8.4f accuracy:%5.2f',
            epoch, torch.tensor(loss_collection).mean().item(),
            100 * torch.tensor(accuracy_collection).mean().item())
    lg.info('Epoch:%02d a:%s b:%s', epoch,
            f.a.data, f.b.data)
    # losses.append(loss_collection)


  # lo = np.reshape(losses, (-1,))
  # i = np.arange(0,opt.num_epochs*(opt.num_samples/opt.batch_size))
  # lo = lo
  # print(lo)
  # print(opt.a)
  # print(opt.b)
  # plt.plot(list(i), lo)
  # plt.ylabel('Losses')
  # plt.xlabel('No of Data')
  # plt.show()


  import PyGnuplot as pg
  import os

  prefix = os.path.join(os.getenv('HOME'), 'tmp')
  dat = os.path.join(prefix, 'optimizing_quadratic_equation.dat')
  png = os.path.join(prefix, 'optimizing_quadratic_equation.png')
  fmt = "'%s' u %s t '%s' lc rgb '%s' lw 2 w %s"

  np.savetxt(dat, list(zip(range(len(loss_collection)),
                           loss_collection,
                           accuracy_collection)))

  pg.c('set term png size 1280,800')
  pg.c('set output "%s"' % (png,))
  pg.c('plot %s' % ','.join(
    fmt % lst for lst in (
      (dat, '1:2', 'Loss', 'red', 'lines'),
      (dat, '1:3', 'Accuracy', 'dark-green', 'lines')
    )
  ))
  # pg.png(png)

  lg.info('Plot: saved in %s', png)