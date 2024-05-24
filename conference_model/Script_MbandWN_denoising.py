# -*- coding: utf-8 -*-
"""
设置多组信号训练
阈值设置多种0.4-1.1，
平均的RMS，2级分解.

"""
import scipy.io

'''
添加100次噪声的结果平均，不是信号分段100个
'''

# Usual packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000
from dataset import LoadSig_N as LoadSig
from dataset import Normalize
import torch
import torch.nn as nn
from utils.torch_random_seed import seed_torch
from scipy import interpolate
import seaborn as sns
from conference_model.WADNet import WADNet,CoeffLoss
from utils.utils_ae import lr_scheduler_choose
from matplotlib.colors import ListedColormap
from conference_model.Script_MbandWN_frequencyanomaly_validation import CoeffLoss,RecLoss,Two_interplate,get_alpha_blend_cmap,coefficient_visul_batch,coefficient_visul,coefficient_visul_pywt,fft_fignal_plot,plot_box
from utils.kernel_selector import Kernel_selector

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# import pywt
seed_torch(666)
# Load a toy time series data to run DeSPAWN
# signal: batch_size*channel*length*1
# 频率在2000Hz正弦波动信号
#100个数据，
Normalization = Normalize()
signal_selection=2# 1变频信号 2: wavelab 生成的piece-regular
if signal_selection==1:
    fs = np.power(2, 13)#8192Hz
    n = np.arange(0, 100 * fs)
    t = n / fs
    generation_signal = np.sin(-400 * np.cos(2 * np.pi * t) + 4000 * np.pi * t)

    length_noise = n.shape[0]
    # noise = np.random.normal(loc=0,scale=1,size=length_noise)
    # signalT_N = generation_signal + 0.5*noise

    noise = np.random.normal(loc=0,scale=0.5,size=length_noise)
    signalT_N = generation_signal + noise

    signalT = generation_signal
    lTrain = int(8192/4)
    data_number = int(length_noise/lTrain)

    signalT = signalT.reshape((data_number,lTrain))
    signalT = signalT[:, np.newaxis, :, np.newaxis]
    signal = signalT[:int(data_number*0.75), :, :, :]
    signal_test = signalT[int(data_number*0.75):, :, :, :]

    signalT_N = signalT_N.reshape((data_number, lTrain))
    signalT_N = signalT_N[:, np.newaxis, :, np.newaxis]
    signal_N = signalT_N[:int(data_number * 0.75), :, :, :]
    signal_test_N = signalT_N[int(data_number * 0.75):, :, :, :]
    plt.figure(figsize=(50,8))
    plt.plot(np.squeeze(signal[0,:,:,:]))
    plt.show()

    plt.figure(figsize=(50,8))
    plt.plot(np.squeeze(signal_N[0,:,:,:]))
    plt.show()

else:
    signal = scipy.io.loadmat('designal.mat')['signal']
    signal = signal[:, np.newaxis, :, np.newaxis]
    # signal = np.repeat(signal, 100, axis=0)
    noise = np.random.normal(loc=0, scale=3.0, size=signal.shape)
    signal_N = signal+noise

    signal_test = np.repeat(signal,100,axis=0)
    noise_test = np.random.normal(loc=0, scale=3.0, size=signal_test.shape)
    signal_test_N = signal_test + noise_test

    Normalization = '1-1'
    Nor=Normalize(Normalization)
    plt.figure(figsize=(50, 8))
    plt.plot(np.squeeze(Nor(signal[0, :, :, :])))
    plt.show()

    plt.figure(figsize=(50, 8))
    plt.plot(np.squeeze(Nor(signal_N[0, :, :, :])))
    plt.show()

lossCoeff_setting = True
coeffLoss=CoeffLoss(lossCoeff_setting)
recLoss=RecLoss()
epochs =1500 #1500
# generates model: model outputs the reconstructed signals, the loss on the wavelet coefficients and wavelet coefficients
m=[3,3]
k=[2,2]
# 2band 4
# 3band 2
# 4band 4
# 5band 4
level=len(m)
print('{} band decomposition, level is {}'.format(m[0],level))
KI = Kernel_selector(band=3)
visualization_kernel= False
mode = 'WPT'
device='cuda'
model = WADNet(kernelInit=2, m=m, k=k, level=level, kernelsConstraint='KIPL', mode=mode, device=device,
                   coefffilter=1.0, realconv=True, initHT=0.1,initHT1=2.0, kernTrainable=True, threshold=True, t=20.0)
#CF KI PL Free
model.cuda()
Normalization = '1-1'
opt = torch.optim.Adam(params=model.parameters() ,lr=3e-2, betas=(0.9,0.999), eps=1e-07)
# opt = torch.optim.SGD(params=model.parameters() ,lr=0.001, momentum=0.9)
lr_scheduler = lr_scheduler_choose(lr_scheduler_way='step',optimizer=opt,steps='500,800',gamma=0.1)#'800,1300,1400'
#
# kernels = model.MBAND
# kernellen = m[0] * k[0]
#
# print('threshold before optimizaiton')
# print('左侧内侧')
# print(model.thre.l)
# print('左侧带通范围')
# print(model.thre.l1)
# print('右侧内侧')
# print(model.thre.r)
# print('右侧带通范围')
# print(model.thre.r1)
#
# for i in range(m[0]):
#     kerneltemp = kernels[0].DeFilter[i].kernel  # 方便的获取滤波器系数
#     print('DE filter')
#     print(torch.sum(kerneltemp))
#     print(kerneltemp.detach().cpu().numpy().reshape(kernellen))
# print("TRAINING STAGE")

train_iter =LoadSig(signal,signal_N,Normalization=Normalization) #用污染的数据训练
for epoch in range(epochs):
    running_loss = 0.0
    all_number = 0
    model.train()
    # train
    for i, [sig,_] in enumerate(train_iter):
        if epoch%300==0 and i==0:
            print('current lr: {}'.format(lr_scheduler.get_lr()))
        opt.zero_grad()
        sig = sig.cuda()
        sigPred, coeff, loss2, hard_threshold_max, visualization_coeff_after_threshold = model(sig)
        loss1 = coeffLoss(coeff)
        r = recLoss(sig, sigPred)
        # loss =  r #重构
        # loss = r + 1.0*loss1 #重构+稀疏
        # loss = 0.5*r + 1.0*loss2 #重构+正交
        # loss = loss1 #稀疏
        # loss = loss1+loss2 #稀疏+正交
        # loss = loss2 #正交约束
        # loss = 0.5*r +0.3*loss1 +1.0*loss2 # 重构+稀疏+正交
        # loss = r +0.3*loss1 +0.3*loss2 # 重构+稀疏+正交
        loss = r + 1.0 * loss1 - 0.1 * hard_threshold_max + 1.0 * loss2  # 重构 + 稀疏 + 阈值带通区间最小化 + 滤波器约束
        loss.backward()
        opt.step()
        running_loss += r.item()*sig.size()[0]
        all_number += sig.size()[0]
    lr_scheduler.step()
    if epoch%300==0:
        print('[epoch:%d/%d] loss: %.3f'
          % (epoch+1, epochs, running_loss/all_number))

train_iter =LoadSig(signal,signal_N,Normalization=Normalization) #用污染的数据训练
model.eval()
print('training set')
for ht in np.arange(0,1.0,0.1):
    denoise_loss = 0.0
    all_number = 0
    for i, [sig, sig_N] in enumerate(train_iter):
        sig = sig.cuda()
        sig_N = sig_N.cuda()
        sigPred, coeff, loss2, hard_threshold_max, visualization_coeff_after_threshold = model(sig_N,noDHT=False,ht=ht)
        if i ==0:
            plt.figure(figsize=(50, 8))
            plt.plot(torch.squeeze(sigPred[0]).detach().cpu().numpy())
            plt.title('trainingset ht is {}'.format(ht))
            plt.show()
        r = recLoss(sig, sigPred)#降噪后信号与原始信号的差值
        denoise_loss += r.item()*sig.size()[0]
        all_number += sig.size()[0]
    denoise_loss = denoise_loss/all_number
    print('loss is {}, ht is {}'.format(denoise_loss,ht))

print('test set')
train_iter =LoadSig(signal_test,signal_test_N,Normalization=Normalization)
model.eval()
for ht in np.arange(0,1.0,0.1):
    denoise_loss = 0.0
    all_number = 0
    for i, [sig, sig_N] in enumerate(train_iter):
        sig = sig.cuda()
        sig_N = sig_N.cuda()
        sigPred, coeff, loss2, hard_threshold_max, visualization_coeff_after_threshold = model(sig_N, noDHT=False,
                                                                                               ht=ht)
        r = recLoss(sig, sigPred)#降噪后信号与原始信号的差值 [:,:,100:4000]
        denoise_loss += r.item()*sig.size()[0]
        all_number += sig.size()[0]
    denoise_loss = denoise_loss/all_number
    print('loss is {}, ht is {}'.format(denoise_loss,ht))

# coefficient_visul(coeff,mode=mode,m=m)#,prefined_sort=[1,2,0,3]
# print('*'*6+'pywt 可视化'+'*'*6)
# max_level=5
# wp = pywt.WaveletPacket(data=signal.reshape(lTrain), wavelet='db4', mode='symmetric',maxlevel=max_level)
# node_list = [node.path for node in wp.get_level(max_level, 'freq')]#'natural' 'freq'
# coeff_pywt =[wp[i].data for i in node_list]
# coefficient_visul_pywt(coeff_pywt)

# # Examples for plotting the model outputs and learnings
# indPlot = 0
# out=[]
# outTe=[]
# print("TESTING STAGE")
# model.eval()
# print('*'*6+'threshold in first layer'+'*'*6)
# print(threshold_list[0].left)
# print(threshold_list[0].right)
#
# print('*'*6+'After traning'+'*'*6)
# for i in range(m[0]):
#     kerneltemp = kernels[0].DeFilter[i].kernel #方便的获取滤波器系数
#     print('DE filter')
#     print(torch.sum(kerneltemp))
#     print(kerneltemp.detach().cpu().numpy().reshape(kernellen))
#     # kerneltemp = kernels[0].ReFilter[i].kernel  # 方便的获取滤波器系数
#     # print('Corresponding RE filter')
#     # print(torch.sum(kerneltemp))
#     # print(kerneltemp.detach().cpu().numpy().reshape(kernellen))
#
#     if visualization_kernel:
#         fft_fignal_plot(kerneltemp.detach().cpu().numpy().reshape(kernellen),fs=2,window=kernellen)


#
# for i, sig in enumerate(train_iter):
#     sig = sig.cuda()
#     out = model(sig)  # [sigPred, vLossCoeff, g_last, hl]
#
# # Test part of the signal
# test_iter=LoadSig(signal_test)
# for i, sig in enumerate(test_iter):
#     sig = sig.cuda()
#     outTe = model(sig)
#      # print(outTe[0].squeeze())



# fig = plt.figure(num=1)
# fig.clf()
# # fig.suptitle('SUP Title')
# ax = fig.add_subplot(1,1,1)
# ax.title.set_text(mode)
# ax.plot(np.arange(signal.shape[2]),signal[indPlot,0,:,0])
# ax.plot(np.arange(signal.shape[2]),out[0][indPlot,0,:,0].detach().cpu().numpy())
#
# ax.plot(np.arange(signal.shape[2],signalT.shape[2]),signalT[indPlot,0,lTrain:,0].squeeze())
# ax.plot(np.arange(signal.shape[2],signalT.shape[2]),outTe[0][indPlot,0,:,0].detach().cpu().numpy())
# ax.legend(['Train Original','Train Reconstructed','Test Original', 'Test Reconstructed'])
# # ax = fig.add_subplot(2,2,3)


# idpl = 0
# # for e,o in enumerate(out[2:]):
# #     ax.boxplot(np.abs(np.squeeze(o[indPlot,:,:,:].detach().numpy())), positions=[e], widths=0.8)
# # ax.set_xlabel('Decomposition Level')
# # ax.set_ylabel('Coefficient Distribution')
# trainYLim = ax.get_ylim()
# trainXLim = ax.get_xlim()
# ax = fig.add_subplot(2,2,4)
# idpl = 0
# for e,o in enumerate(outTe[2:]):
#     # print(o.shape[2])
#     if o.shape[2]>1:
#         ax.boxplot(np.abs(np.squeeze(o[indPlot,:,:,:])), positions=[e], widths=0.8)
#     else:
#         ax.plot(e,np.abs(np.squeeze(o[indPlot,:,:,:])),'o',color='k')
# ax.set_xlabel('Decomposition Level')
# ax.set_ylabel('Coefficient Distribution')
# ax.set_ylim(trainYLim)
# ax.set_xlim(trainXLim)
# fig.show()