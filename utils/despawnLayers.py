# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import numpy as np
from utils.base_convolution import Conv2d,ConvTranspose2d
class Kernel(nn.Module):
    def __init__(self, kernelInit=8, trainKern=True,device ='cuda', **kwargs):
        super(Kernel, self).__init__()
        self.trainKern  = trainKern
        self.kernel=None
        if isinstance(kernelInit,int):
            self.kernelSize = kernelInit
            self.kernel = Parameter(torch.randn(1,1,self.kernelSize, 1,dtype=torch.float32,device=device),requires_grad=True,)
            # nn.init.normal_(self.kernel, mean=0,std=1)
            nn.init.xavier_normal_(self.kernel, gain=1.0)  # 无ReLU时候使用 可以替换为uniform
        else:
            self.kernelSize = len(kernelInit)
            self.kernel = Parameter(torch.tensor(kernelInit,dtype=torch.float32,device=device).reshape(1,1,self.kernelSize,1),requires_grad=True)

    def forward(self):
        return self.kernel


class LowPassWave(nn.Module):
    """
    Layer that performs a convolution between its two inputs with stride (2,1)
    """
    def __init__(self, kernel, device ='cuda',**kwargs):
        super(LowPassWave, self).__init__()
        self.kernel=kernel
        self.device = device
        self.ConvLayer=Conv2d(in_channels=1,out_channels=1,kernel=self.kernel,stride=(2, 1),device =self.device)

    def forward(self, input):
        return self.ConvLayer(input)

class HighPassWave(nn.Module):
    """
    Layer that performs a convolution between its two inputs with stride (2,1).
    Performs first the reverse alternative flip on the second inputs
    """
    def __init__(self,kernel,device ='cuda', **kwargs):
        super(HighPassWave, self).__init__()
        self.kern=kernel
        self.device = device

        self.qmfFlip = torch.reshape(Variable(torch.tensor([(-1) ** (i) for i in range(self.kern.shape[2])], dtype=torch.float32,device =self.device),
                                              requires_grad=True),(1, 1, -1, 1))
        # print(self.qmfFlip)
        self.kernel = torch.multiply(torch.flip(self.kern, [2]), self.qmfFlip)

        self.ConvLayer=Conv2d(in_channels=1,out_channels=1,kernel=self.kernel,stride=(2, 1),device =self.device)

    def forward(self, input):
        return self.ConvLayer(input)

class LowPassTrans(nn.Module):
    """
    Layer that performs a convolution transpose between its two inputs with stride (2,1).
    The third input specifies the size of the reconstructed signal (to make sure it matches the decomposed one)
    """
    def __init__(self, kernel,device ='cuda',**kwargs):
        super(LowPassTrans, self).__init__()
        self.kernel = kernel
        self.device = device
        self.ConvTransLayer = ConvTranspose2d(in_channels=1, out_channels=1, kernel=self.kernel, stride=(2, 1),device =self.device)

    def forward(self, input, out_size):
        return self.ConvTransLayer(input,out_size)

class HighPassTrans(nn.Module):
    """
    Layer that performs a convolution transpose between its two inputs with stride (2,1).
    Performs first the reverse alternative flip on the second inputs
    The third input specifies the size of the reconstructed signal (to make sure it matches the decomposed one)
    """
    def __init__(self, kernel,device ='cuda',**kwargs):
        super(HighPassTrans, self).__init__()
        self.kern = kernel
        self.device = device

        self.qmfFlip = torch.reshape(Variable(torch.tensor([(-1) ** i for i in range(self.kern.shape[2])],device =self.device, dtype=torch.float32),
                                              requires_grad=True), (1, 1, -1, 1))
        # print(self.qmfFlip)
        self.kernel = torch.multiply(torch.flip(self.kern, [2]), self.qmfFlip)

        self.ConvTransLayer = ConvTranspose2d(in_channels=1, out_channels=1, kernel=self.kernel, stride=(2, 1),device =self.device)

    def forward(self, input, out_size):
        return self.ConvTransLayer(input,out_size)

class HardThresholdAssym(nn.Module):
    """
    Learnable Hard-thresholding layers
    """
    def __init__(self, init=None, trainBias=True, **kwargs):
        super(HardThresholdAssym, self).__init__()
        if isinstance(init,float) or isinstance(init,int):
            self.thrP = Parameter(torch.tensor(init).reshape(1,1,1,1),requires_grad=trainBias)
            self.thrN = Parameter(torch.tensor(init).reshape(1,1,1,1),requires_grad=trainBias)
        else:
            self.thrP = Parameter(torch.ones(1, 1, 1, 1), requires_grad=trainBias)
            self.thrN = Parameter(torch.ones(1, 1, 1, 1), requires_grad=trainBias)
        self.trainBias = trainBias

    def forward(self, input):

        return torch.multiply(input,torch.sigmoid(10*(input-self.thrP))+torch.sigmoid(-10*(input+self.thrN)))


if __name__ == '__main__':
    inputSize=10
    kernelInit = np.array([-0.010597401785069032, 0.0328830116668852, 0.030841381835560764, -0.18703481171909309,
                           -0.027983769416859854, 0.6308807679298589, 0.7148465705529157, 0.2303778])
    signal = torch.Tensor(range(16))
    # signal = torch.Tensor(range(1024)).reshape(1, 1, 1024, 1) - 512.0
    signallen = signal.size(0)
    signal=signal.view(1, 1, signallen, 1)
    kernel=Kernel(kernelInit, trainKern=True)
    kern = kernel()
    size=signal.shape
    #分解
    a1 = LowPassWave(kernel=kern)
    d1 = HighPassWave(kernel=kern)
    g = a1(signal)
    h = d1(signal)
    #重构
    ar1 = LowPassTrans(kernel=kern)
    dr1 = HighPassTrans(kernel=kern)
    g = ar1(g,out_size=signal.shape)
    h = dr1(h, out_size=signal.shape)
    g = g+h

    print('------------signal--------------')
    signal=signal.view(signallen).numpy()
    print(signal)
    print('------------reconstruction--------------')
    re_signal=g.view(signallen).detach().numpy()
    print(re_signal)
    print('------------abs_residual--------------')
    residual=np.abs(signal-re_signal)
    print(np.sum(residual))
    #真正的卷积是5，而torch等框架里面的卷积不会对数取负反向.
    # signal = torch.Tensor([1,2]).reshape(1, 1, 2)
    # kernel= torch.Tensor([1, 3]).reshape(1, 1, 2)
    # result=torch.conv1d(signal,kernel,stride=1)
    # print(result)


