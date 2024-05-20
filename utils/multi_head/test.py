import torch
from sklearn import preprocessing
import torch.nn.functional as F
from torch import nn
import os
import numpy as np
def add_record(value, *keys):
    print(keys[0])
    print(keys[1])
    print(value)


def a(x):
    print(id(x))
    x.pop()
    print(x)
    print(id(x))
    x = x + [3]
    print(x)
    print(id(x))

def cross_entropy_error1(y, t):
        if y.ndim == 1:
            t = t.reshape(1, t.size)
            y = y.reshape(1, y.size)
        batch_size = y.shape[0]
        c =np.log(y + 1e-7)
        d = c*t
        a =np.multiply(t,np.log(y + 1e-7))
        b = -np.sum(a) / batch_size
        return b

def cross_entropy_error(y, t):
        if y.ndim == 1:
            t = t.reshape(1, t.size)
            y = y.reshape(1, y.size)
        batch_size = y.shape[0]
        one = torch.ones(y.shape)
        batch_loss = t*torch.log(y) +(one-t)*torch.log(one-y)
        return -batch_loss/batch_size

if __name__ == '__main__':
    l = [1, 2, 3]
    print(id(l))

    msg = "Starting epoch:{:d} | phase: train  | Learning rate: {:d}"
    msg1 ='Starting epoch: {0:<5d}| phase:{1:<10}|⏰: {2:<5d}| Learning rate: {3:.5f}'


    print(msg1.format(1,'train',3,0.666321))
    a =torch.Tensor(3,3)
    print(a)
    b = preprocessing.scale(a)
    print(b)
    c = torch.Tensor(b)
    print(c)
    # add_record(1,2,3)

    print (os.path.dirname('C:\\Windows\\System32\\nvrtc-builtins64_92.dll'))

    input = torch.randn((3, 2), requires_grad=True)
    target = torch.rand((3, 2), requires_grad=False)



    input = np.random.randn(3,2)
    target =  np.random.randn(3, 2)
    loss = cross_entropy_error1(input, target)
    loss = nn.CrossEntropyLoss()
    input = torch.randn(3, 5, requires_grad=True)
    target = torch.empty(3, dtype=torch.long).random_(5)
    output = loss(input, target)

    input = torch.randn(3, 5, requires_grad=True)
    print(input)
    target = input
    a = torch.zeros(1, 5, requires_grad=True)
    b = torch.repeat_interleave(a,3,dim=0)

    b = torch.zeros(3, 5, requires_grad=True)
    a = torch.ones(3, 5, requires_grad=True)
    print(a)
    b =torch.where(a>0.9,a,b)
    b = b*3
    c=  b.sum(dim=1)
    b = 1-b
    print(b)

    a = {'a':torch.Tensor([0,1,3])}

    print(a.values())

