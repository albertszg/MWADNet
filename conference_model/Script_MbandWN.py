# -*- coding: utf-8 -*-
"""
多带小波网络的简单验证
无需训练，很快
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
import pywt
from conference_model.WaveletAnomalyDetectionNet import plot_energydistribution
from conference_model.Script_MbandWN_frequencyanomaly_validation import CoeffLoss,RecLoss,Two_interplate,get_alpha_blend_cmap,coefficient_visul_batch,coefficient_visul,coefficient_visul_pywt,fft_fignal_plot,plot_box
from utils.kernel_selector import Kernel_selector

if __name__ == '__main__':
    seed_torch(666)
    generating_signal = 2

    # Load a toy time series data to run DeSPAWN
    # signal: batch_size*channel*length*1

    if generating_signal == 1:
        # 生成信号
        fs = np.power(2, 13)
        n = np.arange(1, 13 * fs)
        FS_varying = 0.05 * n
        t = n / fs
        # generation_signal = np.cos(np.pi*FS_varying*t)
        generation_signal = np.cos(np.pi * FS_varying * n)

        length_noise = n.shape[0]
        noise = np.random.randn(length_noise)

        signalT = generation_signal
        lTrain = 81920
        signalT = signalT[np.newaxis, np.newaxis, :, np.newaxis]
        signal = signalT[:, :, :lTrain, :]
        signal_test = signalT[:, :, lTrain:, :]
    elif generating_signal == 2:
        #变频 正弦规律变频
        fs = np.power(2, 13)
        n = np.arange(1, 13 * fs) # 13秒 10s给训练集 3s给测试集
        t = n / fs
        generation_signal = np.sin(-400 * np.cos(2 * np.pi * t) + 4000 * np.pi * t)

        length_noise = n.shape[0]
        noise = np.random.normal(loc=0, scale=0.2, size=length_noise)

        signalT = generation_signal  # + noise
        lTrain = fs*4
        signalT = signalT[np.newaxis, np.newaxis, :, np.newaxis]
        signal = signalT[:, :, :lTrain, :] # 训练集 训练集和测试集长度不等 10s
        signal_test = signalT[:, :, lTrain:, :] #测试集 3s 8192个点*3s
    elif generating_signal == 3:
        #变频 直线变频 好像有错，不对劲
        fs = np.power(2, 13)
        n = np.arange(1, 13 * fs) # 13秒 10s给训练集 3s给测试集
        t = n / fs
        generation_signal = np.cos(np.pi*0.05*t*t)

        length_noise = n.shape[0]
        noise = np.random.normal(loc=0, scale=0.2, size=length_noise)

        signalT = generation_signal  # + noise
        lTrain = fs*10
        signalT = signalT[np.newaxis, np.newaxis, :, np.newaxis]
        signal = signalT[:, :, :lTrain, :] # 训练集 训练集和测试集长度不等 10s
        signal_test = signalT[:, :, lTrain:, :] #测试集 3s 8192个点*3s
    else:
        raise Exception('Not implemented')


    lossCoeff_setting = True
    coeffLoss=CoeffLoss(lossCoeff_setting)
    recLoss=RecLoss()

    epochs =1000#默认2000
    # generates model: model outputs the reconstructed signals, the loss on the wavelet coefficients and wavelet coefficients
    # m=[5,5,5]
    # k=[4,4,4]
    m=[3,3,3]
    # m=[5,5,5,5,5]
    k=[2,2,2]

    level = len(m)
    mode = 'WPT'
    visualization_kernel = False
    device = 'cuda'
    model = WADNet(kernelInit=3, m=m, k=k, level=level, kernelsConstraint='KIPL', mode=mode, device=device,
                   coefffilter=1.0, realconv=True, initHT=0.1, initHT1=2.0, kernTrainable=True, threshold=True, t=10.0)
    model.cuda()

    opt = torch.optim.Adam(params=model.parameters() ,lr=3e-2, betas=(0.9,0.999), eps=1e-07)
    lr_scheduler = lr_scheduler_choose(lr_scheduler_way='step',optimizer=opt,steps='800,1300,1400',gamma=0.1)

    kernels = model.MBAND
    kernellen=m[0]*k[0]

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
    for epoch in range(epochs):
        running_loss = 0.0
        model.train()
        # train
        for i, sig in enumerate(train_iter):
            print('current lr: {}'.format(lr_scheduler.get_lr()))
            opt.zero_grad()
            sig = sig.cuda()
            sigPred, coeff, loss2, hard_threshold_max, visualization_coeff_after_threshold = model(sig)
            loss1 = coeffLoss(coeff)
            r = recLoss(sig, sigPred)  # 重构loss
            # loss = r  # 仅重构
            # loss = r + 10.0 * loss2 #重构 + 滤波器约束
            # loss = r +  1.0*loss1 #重构 + 稀疏
            # loss = r +5.0*loss1 +1.0*loss2 #重构+稀疏+滤波器约束
            # loss = r - 30.0*sigmoid(hard_threshold_max)# 重构 + 阈值带通区间最小化
            # loss = r-0.5*(hard_threshold_max)# 重构 + 阈值带通区间最小化
            # loss = r + 1.0*loss1 - 2.0*hard_threshold_max# 重构 + 稀疏 + 阈值带通区间最小化
            # loss = r + 1.0*loss2 - 5.0*hard_threshold_max# 重构 + 滤波器 + 阈值带通区间最小化/
            loss = 2.0*r + 2.0 * loss1 - 0.2 * hard_threshold_max + 1.0 * loss2  # 重构 + 稀疏 + 阈值带通区间最小化 + 滤波器约束
            loss.backward()
            opt.step()
            running_loss += r.item()
            lr_scheduler.step()
        print('[epoch:%d/%d] loss: %.3f'
              % (epoch + 1, epochs, running_loss / len(train_iter)))

    coefficient_visul(visualization_coeff_after_threshold, mode=mode, m=m,title='train')  # ,prefined_sort=[1,2,0,3]

    print('*'*6+'pywt 可视化'+'*'*6)
    max_level=5
    wp = pywt.WaveletPacket(data=signal.reshape(lTrain), wavelet='db4', mode='symmetric',maxlevel=max_level)
    node_list = [node.path for node in wp.get_level(max_level, 'freq')]#'natural' 'freq'
    coeff_pywt =[wp[i].data for i in node_list]
    coefficient_visul_pywt(coeff_pywt,title='pywt')


    # Examples for plotting the model outputs and learnings
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

    for i, sig in enumerate(train_iter):
        sig = sig.cuda()
        out = model(sig)  # [sigPred, vLossCoeff, g_last, hl]
    coefficient_visul(out[1], mode=mode, m=m, title='Test Signal A of model')  # ,prefined_sort=[1,2,0,3]
    plot_box(out[1], batchidx=0, title='Test Signal A of model (before threshold)', mode='WPT', m=m)
    plot_box(out[-1], batchidx=0, title='Test Signal A of model (after threshold)')

    # Test part of the signal
    test_iter=LoadSig(signal_test)
    for i, sig in enumerate(test_iter):
        sig = sig.cuda()
        outTe = model(sig)
         # print(outTe[0].squeeze())
    coefficient_visul(outTe[1], mode=mode, m=m, title='Test Signal B of model')  # ,prefined_sort=[1,2,0,3]
    plot_box(outTe[1], batchidx=0, title='Test Signal B of model (before threshold)', mode='WPT', m=m)
    plot_box(outTe[-1], batchidx=0, title='Test Signal B of model (after threshold)')

    print('*' * 6 + 'pywt 可视化' + '*' * 6)
    max_level = 5
    wp = pywt.WaveletPacket(data=signal_test.reshape(13*fs-lTrain-1), wavelet='db4', mode='symmetric', maxlevel=max_level)
    node_list = [node.path for node in wp.get_level(max_level, 'freq')]  # 'natural' 'freq'
    coeff_pywt = [wp[i].data for i in node_list]
    coefficient_visul_pywt(coeff_pywt, title='pywt')

    fig = plt.figure(num=1)
    fig.clf()
    # fig.suptitle('SUP Title')
    ax = fig.add_subplot(1,1,1)
    ax.title.set_text(mode)
    ax.plot(np.arange(signal.shape[2]),signal[indPlot,0,:,0])
    ax.plot(np.arange(signal.shape[2]),out[0][indPlot,0,:,0].detach().cpu().numpy())

    ax.plot(np.arange(signal.shape[2],signalT.shape[2]),signalT[indPlot,0,lTrain:,0].squeeze())
    ax.plot(np.arange(signal.shape[2],signalT.shape[2]),outTe[0][indPlot,0,:,0].detach().cpu().numpy())
    ax.legend(['Train Original','Train Reconstructed','Test Original', 'Test Reconstructed'])

    reconstruction_error = np.mean((signal[indPlot,0,:,0] - out[0][indPlot,0,:,0].detach().cpu().numpy()) ** 2)
    print('重构误差 训练集 {}'.format(reconstruction_error))
    error_B = np.mean((signalT[indPlot,0,lTrain:,0] - outTe[0][indPlot,0,:,0].detach().cpu().numpy()) ** 2)
    print('重构误差 测试集 {}'.format(error_B))

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
    fig.show()