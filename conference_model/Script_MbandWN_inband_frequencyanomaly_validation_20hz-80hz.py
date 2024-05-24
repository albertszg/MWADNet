# -*- coding: utf-8 -*-
"""
在同一频带内的信号如何靠稀疏来识别
10Hz,30Hz，采频500Hz，
"""
# Usual packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataset import LoadSig
import torch
import torch.nn as nn
from utils.torch_random_seed import seed_torch
from scipy import interpolate
import seaborn as sns

from conference_model.WADNet import WADNet,CoeffLoss
from utils.utils_ae import lr_scheduler_choose
from matplotlib.colors import ListedColormap
from conference_model.Script_MbandWN_frequencyanomaly_validation import CoeffLoss,RecLoss,Two_interplate,get_alpha_blend_cmap,coefficient_visul_batch,coefficient_visul,coefficient_visul_pywt,fft_fignal_plot,plot_box
import pywt
from conference_model.WaveletAnomalyDetectionNet import plot_energydistribution
if __name__ == '__main__':
    seed_torch(666)
    generating_signal = 4

    # Load a toy time series data to run DeSPAWN
    # signal: batch_size*channel*length*1
    fs = 500
    n = np.arange(0, 4 * fs)  # 2s时间，1s训练，1s测试
    t = n / fs
    signal_1 = np.sin(20 * 2 * np.pi * t)
    signal_2 = np.cos(40 * 2 * np.pi * t)

    fft_fignal_plot(signal_1, fs, 1000, plot=True,title='signal1')
    fft_fignal_plot(signal_2, fs, 1000, plot=True,title='signal2')

    lTrain = 1000

    signalT1 = signal_1[np.newaxis, np.newaxis, :, np.newaxis]
    signalT2 = signal_2[np.newaxis, np.newaxis, :, np.newaxis]

    signal = signalT1[:, :, :lTrain, :]# 正常训练集
    signal_A = signalT1[:, :, lTrain:, :]#正常 测试集
    signal_test = signalT2[:, :, :lTrain, :] #异常测试
    signal_B = signalT2[:, :, lTrain:, :] #异常测试


    lossCoeff_setting = True
    coeffLoss = CoeffLoss(lossCoeff_setting)
    recLoss = RecLoss()

    epochs = 1

    m = [2]
    k = [4]
    level = len(m)
    mode = 'WPT'
    visualization_kernel = True
    device = 'cuda'
    # Kernel_I = Kernel_selector(band=3)
    model = WADNet(kernelInit=2, m=m, k=k, level=level, kernelsConstraint='KIPL', mode=mode, device=device,
                   coefffilter=1.0, realconv=True, initHT=0.1,initHT1=2.0, kernTrainable=True, threshold=True, t=20.0)
    model.cuda()

    opt = torch.optim.Adam(params=model.parameters(), lr=3e-2, betas=(0.9, 0.999), eps=1e-07)
    lr_scheduler = lr_scheduler_choose(lr_scheduler_way='step', optimizer=opt, steps='800,1300,1400', gamma=0.1)

    kernels = model.MBAND
    kernellen = m[0] * k[0]


    print('threshold before optimizaiton')
    print('左侧内侧')
    print(model.thre.l)
    print('左侧带通范围')
    print(model.thre.l1)
    print('右侧内侧')
    print(model.thre.r)
    print('右侧带通范围')
    print(model.thre.r1)

    for i in range(m[0]):
        kerneltemp = kernels[0].DeFilter[i].kernel  # 方便的获取滤波器系数
        print('DE filter')
        print(torch.sum(kerneltemp))
        print(kerneltemp.detach().cpu().numpy().reshape(kernellen))
    print("TRAINING STAGE")
    train_iter = LoadSig(signal)  # batch_size * c * h * w
    sigmoid = torch.nn.Sigmoid()
    for epoch in range(epochs):
        running_loss = 0.0
        model.train()
        # train
        for i, sig in enumerate(train_iter):
            print('current lr: {}'.format(lr_scheduler.get_lr()))
            opt.zero_grad()
            sig = sig.cuda()
            sigPred, coeff, loss2, hard_threshold_max,visualization_coeff_after_threshold = model(sig)
            loss1 = coeffLoss(coeff)
            r = recLoss(sig, sigPred)  # 重构loss
            # loss = r  # 仅重构
            # loss = r + 10.0 * loss2 #重构 + 滤波器约束
            # loss = r +  1.0*loss1 #重构 + 稀疏
            # loss = r +loss1 +10.0*loss2 #重构+稀疏+滤波器约束
            # loss = r - 30.0*sigmoid(hard_threshold_max)# 重构 + 阈值带通区间最小化
            # loss = r-0.5*(hard_threshold_max)# 重构 + 阈值带通区间最小化
            # loss = r + 1.0*loss1 - 2.0*hard_threshold_max# 重构 + 稀疏 + 阈值带通区间最小化
            # loss = r + 1.0*loss2 - 5.0*sigmoid(hard_threshold_max)# 重构 + 滤波器 + 阈值带通区间最小化
            loss = r + 1.0*loss1 - 0.1*hard_threshold_max + 1.0 * loss2# 重构 + 稀疏 + 阈值带通区间最小化 + 滤波器约束
            loss.backward()
            opt.step()
            running_loss += r.item()
            lr_scheduler.step()
        print('[epoch:%d/%d] loss: %.3f'
              % (epoch + 1, epochs, running_loss / len(train_iter)))

    coefficient_visul(visualization_coeff_after_threshold, mode=mode, m=m,title='train')  # ,prefined_sort=[1,2,0,3]

    indPlot = 0
    out = []
    outTe = []
    print("TESTING STAGE")
    model.eval()
    print('threshold after optimizaiton')
    print('左侧内侧')
    print(model.thre.l)
    print('左侧带通范围')
    print(model.thre.l1)
    print('右侧内侧')
    print(model.thre.r)
    print('右侧带通范围')
    print(model.thre.r1)

    print('*' * 6 + 'After traning' + '*' * 6)
    for i in range(m[0]):
        kerneltemp = kernels[0].DeFilter[i].kernel  # 方便的获取滤波器系数
        print('DE filter')
        print(torch.sum(kerneltemp))
        print(kerneltemp.detach().cpu().numpy().reshape(kernellen))
        # kerneltemp = kernels[0].ReFilter[i].kernel  # 方便的获取滤波器系数
        # print('Corresponding RE filter')
        # print(torch.sum(kerneltemp))
        # print(kerneltemp.detach().cpu().numpy().reshape(kernellen))
        if visualization_kernel:
            fft_fignal_plot(kerneltemp.detach().cpu().numpy().reshape(kernellen), fs=2, window=kernellen)

    test_iter_A = LoadSig(signal_A)
    for i, sig in enumerate(test_iter_A):
        sig = sig.cuda()
        out = model(sig)  # [sigPred, vLossCoeff, g_last, hl]
    coefficient_visul(out[1], mode=mode, m=m, title='Test Signal A of model')  # ,prefined_sort=[1,2,0,3]
    plot_box(out[1], batchidx=0, title='Test Signal A of model (before threshold)',mode = 'WPT', m=m)
    plot_box(out[-1], batchidx=0, title='Test Signal A of model (after threshold)')
    # Test part of the signal
    test_iter = LoadSig(signal_B)
    for i, sig in enumerate(test_iter):
        sig = sig.cuda()
        outTe = model(sig)
        # print(outTe[0].squeeze())
    coefficient_visul(outTe[1], mode=mode, m=m,title='Test Signal B of model')  # ,prefined_sort=[1,2,0,3]
    plot_box(outTe[1], batchidx=0, title='Test Signal B of model (before threshold)',mode = 'WPT', m=m)
    plot_box(outTe[-1], batchidx=0, title='Test Signal B of model (after threshold)')
    # signal = signalT1[:, :, :lTrain, :]  # 正常训练集
    # signal_A = signalT1[:, :, lTrain:, :]  # 正常 测试集
    # signal_test = signalT2[:, :, :lTrain, :]  # 异常测试
    # signal_B = signalT2[:, :, lTrain:, :]  # 异常测试

    fig = plt.figure(num=1)
    fig.clf()
    ax = fig.add_subplot(1, 1, 1)
    ax.title.set_text('Test A reconstruction')
    ax.plot(np.arange(signal_A.shape[2]), signal_A[indPlot, 0, :, 0], '#1f77b4')
    ax.plot(np.arange(signal_A.shape[2]), out[0][indPlot, 0, :, 0].detach().cpu().numpy(),'#ff7f0e')

    ax.legend(['Reconstructed signal A','Signal A'])
    fig.show()
    reconstruction_error = np.mean((signal_A[indPlot, 0, :, 0] - out[0][indPlot, 0, :, 0].detach().cpu().numpy()) ** 2)
    print('重构误差 A {}'.format(reconstruction_error))

    fig = plt.figure(num=1)
    fig.clf()
    ax = fig.add_subplot(1, 1, 1)
    x =np.arange(signal_A.shape[2], signalT2.shape[2])
    y =signal_test[indPlot, 0,:, 0].squeeze()
    ax.plot(x,y, '#2ca02c')#signal B
    x1 = np.arange(signal.shape[2], signalT2.shape[2])
    y1 = outTe[0][indPlot, 0, :, 0].detach().cpu().numpy()
    # ax = fig.add_subplot(1, 1, 1)
    ax.plot(x1,y1,'#ff7f0e')
    ax.legend(['Signal B', 'Reconstructed Signal B'])
    fig.show()
    error_B = np.mean((y - y1) ** 2)
    print('重构误差 B {}'.format(error_B))


    # ax = fig.add_subplot(2,2,3)
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
