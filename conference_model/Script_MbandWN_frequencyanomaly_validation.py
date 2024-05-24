# -*- coding: utf-8 -*-
"""
频率验证
一层两带分解情况下
共分2个通道，一个低频10Hz，一个高频240Hz，采频500Hz
带间异常检测
此文件构建了基本的可视化函数
"""
# Usual packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize']=(8,6)
from dataset import LoadSig
import torch
import torch.nn as nn
from utils.torch_random_seed import seed_torch
from scipy import interpolate
import seaborn as sns

from utils.utils_ae import lr_scheduler_choose
from matplotlib.colors import ListedColormap
from utils.kernel_selector import Kernel_selector
import pywt
from conference_model.WADNet import WADNet,CoeffLoss
# Set sparsity (dummy) loss:
# the sparsity term has no ground truth => just input an empty numpy array as ground truth (anything would do, in coeffLoss, yTrue is not called)

def plot_box(z,batchidx=0,title=' ',mode='WPT',m=None,fontsize=10):
    #画小波系数的分布盒型图
    # s12=pd.Series(np.squeeze(z[0][0][batchidx,:,:,:].detach().cpu().numpy()))
    # s13=pd.Series(np.squeeze(z[0][1][batchidx,:,:,:].detach().cpu().numpy()))
    # s22=pd.Series(np.squeeze(z[1][0][batchidx,:,:,:].detach().cpu().numpy()))
    # s23=pd.Series(np.squeeze(z[1][1][batchidx,:,:,:].detach().cpu().numpy()))
    # s31=pd.Series(np.squeeze(z[3][0][batchidx,:,:,:].detach().cpu().numpy()))
    # s32=pd.Series(np.squeeze(z[2][0][batchidx,:,:,:].detach().cpu().numpy()))
    # s33=pd.Series(np.squeeze(z[2][1][batchidx,:,:,:].detach().cpu().numpy()))
    # data = pd.DataFrame({"1-2": s12, "1-3": s13, "2-2": s22,"2-3":s23,"3-1":s31,"3-2":s32,"3-3":s33})
    # data.boxplot(grid=False)  # 这里，pandas自己有处理的过程，很方便哦。
    # plt.xlabel("level-band")  # 我们设置横纵坐标的标题。
    # plt.title('coefficient'+title)
    # plt.show()
    # fontsize = 28
    node_number = 1
    for i in m:
        node_number = node_number * i
    z = z[-node_number:]

    data = []
    for i in range(len(z)):
            # data.append(np.squeeze(z[i][j][batchidx,:,:,:].detach().cpu().numpy()))
        data.append(np.abs(np.squeeze(z[i][batchidx,:,:,:].detach().cpu().numpy())))

    # plt.figure(figsize=(10,6))
    plt.figure()
    plt.boxplot(data,patch_artist=True,vert=True,notch=True)

    plt.xlabel('Band node',fontsize=fontsize)
    plt.ylabel('Coefficient distribution',fontsize=fontsize)
    plt.tick_params(labelsize=fontsize)
    plt.tight_layout()
    # plt.title('coefficient'+title)
    # 显示图形
    plt.show()
# Set residual loss:
class RecLoss(nn.Module):
    def __init__(self):
        super(RecLoss, self).__init__()

    def forward(self, yTrue, yPred):
        return torch.mean(torch.abs(yTrue - yPred))


def Two_interplate(coeff, now_len, target_len):
    '''
    :param coeff: numpy
    :param len: int
    :param target_len: int
    :return: coeff with length target_len
    '''

    interplated_signal = coeff
    len_list = []
    len_temp = 2 * now_len
    while len_temp < target_len:
        len_list.append(len_temp)
        len_temp = 2 * len_temp
    len_list.append(target_len)
    x = np.linspace(1, target_len, now_len)  # 初始数目

    for i in range(len(len_list)):
        f = interpolate.interp1d(x, interplated_signal, kind='nearest')
        x = np.linspace(1, target_len, len_list[i])
        interplated_signal = f(x)
    return interplated_signal


def get_alpha_blend_cmap(cmap, alpha):
    cls = plt.get_cmap(cmap)(np.linspace(0, 1, 256))
    cls = (1 - alpha) + alpha * cls
    return ListedColormap(cls)

def plot_energydistribution(z, batchidx=0,title=' ',mode='DWT',m=None):# 数据
    # 画小波系数的能量分布柱状图
    # x_labels = ["1-2", "1-3", "2-2","2-3","3-1","3-2","3-3"]
    #
    # s12=np.sum(z[0][0][batchidx,:,:,:].detach().cpu().numpy()**2)
    # s13=np.sum(z[0][1][batchidx,:,:,:].detach().cpu().numpy()**2)
    # s22=np.sum(z[1][0][batchidx,:,:,:].detach().cpu().numpy()**2)
    # s23=np.sum(z[1][1][batchidx,:,:,:].detach().cpu().numpy()**2)
    # s31=np.sum(z[3][0][batchidx,:,:,:].detach().cpu().numpy()**2)
    # s32=np.sum(z[2][0][batchidx,:,:,:].detach().cpu().numpy()**2)
    # s33=np.sum(z[2][1][batchidx,:,:,:].detach().cpu().numpy()**2)
    #
    # data = np.array([s12,s13,s22,s23,s31,s32,s33])
    # data =data / np.max(data)
    # data = list(data)
    # plt.bar(range(len(data)), data, color='b')
    # # 指定横坐标刻度
    # plt.xticks(range(len(data)), x_labels)
    # plt.xlabel("level-band")  # 我们设置横纵坐标的标题。
    # plt.title(title + 'energy distribution')
    # plt.show()
    # if mode == 'DWT':
    #     data = []
    #     for i in range(len(z)):
    #         for j in range(len(z[i])):
    #             data.append(np.sum(z[i][j][batchidx, :, :, :].detach().cpu().numpy() ** 2))
    #             # data.append(np.average(z[i][j][batchidx, :, :, :].detach().cpu().numpy() ** 2))
    #
    #     data = np.array(data)
    #     # data =data / np.max(data)
    #     data = list(data)
    #     plt.bar(range(len(data)), data, color='b')
    #     # 指定横坐标刻度
    #     # plt.xticks(range(len(data)), x_labels)
    #     plt.xlabel("level-band")  # 我们设置横纵坐标的标题。
    #     plt.title(title + 'energy distribution')
    #     plt.show()
    # elif mode == 'WPT' and isinstance(m, list):
    node_number = 1
    for i in m:
        node_number = node_number * i
    z = z[-node_number:]
    data = []
    for i in range(len(z)):
        data.append(np.sum(z[i][batchidx, :, :, :].detach().cpu().numpy() ** 2))
    data = np.array(data)
    # data =data / np.max(data)
    data = list(data)
    plt.bar(range(len(data)), data, color='b')
    # 指定横坐标刻度
    # plt.xticks(range(len(data)), x_labels)
    plt.xlabel("level-band")  # 我们设置横纵坐标的标题。
    plt.title(title + 'energy distribution')
    plt.show()
    # else:
    #     print('not implemented function')

def coefficient_visul_batch(all_coeff, mode='DWT', m=None, sort='freq', prefined_sort=None,title='coeffifient',batch_idx=0):
    '''
    :param all_coeff: list consis of tensor
    prefined_sort:list [0,2,3,1] dwt后小波的顺序
    :param mode:
        WPT：低频到高频 按节点来排
        DWT：从低频到高频，最后一个是低频，倒数（m-1）个共同组成最后一个M带分解。
            以后每一个m-1
            长度不同的需要复制插值
    :param m:
    :return: 从上至下：高频至低频的时频图
    '''
    node_number = 1
    for i in m:
        node_number = node_number * i
    all_coeff = all_coeff[-node_number:]#[[batch:channel:length:hight] [x] [x] [x] [x][x]] 节点个数个 batch在内部
    # node_number = len(all_coeff)
    if sort == 'freq':
        # https://zhuanlan.zhihu.com/p/528435064
        for lev in range(len(m)):  # 递归推算
            if lev == 0:
                idx = [i for i in range(m[lev])]
            else:
                idx_temp = []
                for i in range(len(idx)):
                    m_lev = m[lev]
                    # 从0开始计数序号则偶不变奇变
                    # if i == len(idx)-1:
                    if i % 2 == 0:  # 偶不变
                        temp = [t for t in range(idx[i] * m_lev, (idx[i] + 1) * m_lev)]
                    else:  # 奇变
                        temp = [t for t in range((idx[i] + 1) * m_lev - 1, idx[i] * m_lev - 1, -1)]
                    idx_temp.extend(temp)
                idx = idx_temp
        assert len(idx) == node_number #获取实际 node 节点的顺序
        coeff = all_coeff[idx[node_number - 1]].detach().cpu().numpy()
        length = coeff.shape[2]
        batch_number = coeff.shape[0]
        coeff = coeff.reshape(batch_number, length)
        coeff = coeff[batch_idx, :].reshape(1, length)# 获取第一个coeff的 并将其性质重置为标准长度
        for i in range(node_number - 2, -1, -1):
            coeff_temp = all_coeff[idx[i]].detach().cpu().numpy().reshape(batch_number, length)
            coeff_temp = coeff_temp[batch_idx,:].reshape(1, length)
            coeff = np.concatenate([coeff, coeff_temp])
    else:
        node_number = 1
        for i in m:
            node_number = node_number * i
        all_coeff = all_coeff[-node_number:]
        # node_number = len(all_coeff)
        coeff = all_coeff[node_number - 1].detach().cpu().numpy()
        length = coeff.shape[2]
        batch_number = coeff.shape[0]
        coeff = coeff.reshape(batch_number, length)
        coeff = coeff[batch_idx, :].reshape(1, length)
        for i in range(node_number - 2, -1, -1):
            coeff_temp = all_coeff[i].detach().cpu().numpy().reshape(batch_number, length)
            coeff_temp = coeff_temp[batch_idx, :].reshape(1, length)
            coeff = np.concatenate([coeff, coeff_temp])

    # ax = sns.heatmap(coeff,cmap='YlGnBu_r')
    # plt.show()
    # ax = sns.heatmap(coeff, cmap=sns.cubehelix_palette(as_cmap=True))#渐变色盘：sns.cubehelix_palette()使用)
    # plt.show()
    # cmap = plt.get_cmap('Set3')
    # ax = sns.heatmap(coeff, cmap=cmap)
    # plt.show()
    ax = sns.heatmap(coeff)  # cmap='YlGnBu_r' 也还可以
    plt.title(title)
    plt.show()

def coefficient_visul(all_coeff, mode='DWT', m=None, sort='freq', prefined_sort=None,title='coeffifient'):
    '''
    :param all_coeff: list consis of tensor
    prefined_sort:list [0,2,3,1] dwt后小波的顺序
    :param mode:
        WPT：低频到高频 按节点来排
        DWT：从低频到高频，最后一个是低频，倒数（m-1）个共同组成最后一个M带分解。
            以后每一个m-1
            长度不同的需要复制插值
    :param m:
    :return: 从上至下：高频至低频的时频图
    '''

    # WPT
    if isinstance(m, list):
        node_number = 1
        for i in m:
            node_number = node_number*i
        all_coeff = all_coeff[-node_number:]
    else:
        node_number = len(all_coeff)
    if sort == 'freq':
        # https://zhuanlan.zhihu.com/p/528435064
        for lev in range(len(m)):  # 递归推算
            if lev == 0:
                idx = [i for i in range(m[lev])]
            else:
                idx_temp = []
                for i in range(len(idx)):
                    m_lev = m[lev]
                    # 从0开始计数序号则偶不变奇变
                    # if i == len(idx)-1:
                    if i % 2 == 0:  # 偶不变
                        temp = [t for t in range(idx[i] * m_lev, (idx[i] + 1) * m_lev)]
                    else:  # 奇变
                        temp = [t for t in range((idx[i] + 1) * m_lev - 1, idx[i] * m_lev - 1, -1)]
                    idx_temp.extend(temp)
                idx = idx_temp
        assert len(idx) == node_number
        coeff = all_coeff[idx[node_number - 1]].detach().cpu().numpy()
        length = coeff.shape[2]
        coeff = coeff.reshape(1, length)
        for i in range(node_number - 2, -1, -1):
            coeff_temp = all_coeff[idx[i]].detach().cpu().numpy().reshape(1, length)
            coeff = np.concatenate([coeff, coeff_temp])
    else:
        if isinstance(m, list):
            node_number = 1
            for i in m:
                node_number = node_number * i
            all_coeff = all_coeff[-node_number:-1]
        else:
            node_number = len(all_coeff)
        coeff = all_coeff[node_number - 1].detach().cpu().numpy()
        length = coeff.shape[2]
        coeff = coeff.reshape(1, length)
        for i in range(node_number - 2, -1, -1):
            coeff_temp = all_coeff[i].detach().cpu().numpy().reshape(1, length)
            coeff = np.concatenate([coeff, coeff_temp])

    # ax = sns.heatmap(coeff,cmap='YlGnBu_r')
    # plt.show()
    # ax = sns.heatmap(coeff, cmap=sns.cubehelix_palette(as_cmap=True))#渐变色盘：sns.cubehelix_palette()使用)
    # plt.show()
    # cmap = plt.get_cmap('Set3')
    # ax = sns.heatmap(coeff, cmap=cmap)
    # plt.show()
    # ax = sns.heatmap(coeff, vmin=-0.5, vmax=0.5)
    plt.title(title)
    # ax = sns.heatmap(coeff, vmin=-1.0, vmax=1.0)  # cmap='YlGnBu_r' 也还可以
    ax = sns.heatmap(coeff,vmin=-2.0, vmax=2.0)  # cmap='YlGnBu_r' 也还可以
    plt.show()


def coefficient_visul_pywt(all_coeff, mode='DWT',title='pywt'):
    '''
    :param all_coeff: list consis of tensor
    :param mode:
        WPT：低频到高频 按节点来排
        DWT：从低频到高频，最后一个是低频，倒数（m-1）个共同组成最后一个M带分解。
            以后每一个m-1
            长度不同的需要复制插值
    :param m:
    :return: 从上至下：高频至低频的时频图
    '''

    node_number = len(all_coeff)
    coeff = all_coeff[node_number - 1]
    length = coeff.shape[0]
    coeff = coeff.reshape(1, length)
    for i in range(node_number - 2, -1, -1):
        coeff_temp = all_coeff[i].reshape(1, length)
        coeff = np.concatenate([coeff, coeff_temp])

    # ax = sns.heatmap(coeff,cmap='YlGnBu_r')
    # plt.show()
    # ax = sns.heatmap(coeff, cmap=sns.cubehelix_palette(as_cmap=True))#渐变色盘：sns.cubehelix_palette()使用)
    # plt.show()
    # cmap = plt.get_cmap('Set3')
    # ax = sns.heatmap(coeff, cmap=cmap)
    # plt.show()
    ax = sns.heatmap(coeff,vmin=-2.0, vmax=2.0)  # cmap='YlGnBu_r' 也还可以
    # ax = sns.heatmap(coeff)  # cmap='YlGnBu_r' 也还可以
    plt.title(title)
    plt.show()


def fft_fignal_plot(signal, fs, window, plot=True,title='spectrum',color='b'):
    sampling_rate = fs
    fft_size = window
    # t = np.arange(0, fft_size/sampling_rate, 1.0 / sampling_rate)
    t = np.arange(0, fft_size, 1)
    mean = signal.mean()
    xs = signal[:fft_size] - mean
    xf = np.fft.rfft(xs) / fft_size
    freqs = np.linspace(0.0, sampling_rate / 2.0, int(fft_size / 2 + 1))
    xfp = np.abs(xf)
    # xfp = 20 * np.log10(np.clip(np.abs(xf), 1e-20, 1e100))
    if plot:
        plt.figure(figsize=(8, 4))
        plt.subplot(211)
        plt.plot(t[:fft_size], xs, color=color)
        plt.xlabel("Time/s")
        plt.ylabel('Amplitude')
        # plt.title("signal")

        plt.subplot(212)
        plt.plot(freqs, xfp, color=color)
        plt.xlabel("Frequency/Hz")
        # 字体FangSong
        plt.ylabel('Amplitude')
        plt.subplots_adjust(hspace=0.4)
        # plt.title(title)
        '''subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
        有六个可选参数来控制子图布局。值均为0~1之间。其中left、bottom、right、top围成的区域就是子图的区域。
        wspace、hspace分别表示子图之间左右、上下的间距。实际的默认值由matplotlibrc文件控制的。
        '''
        plt.show()

def plot_experiment(dataset,title='signal',linewidth=3,fontsize=32,model=None,indPlot=0,m=[2],color='b',color2='#ff7f0e'):

    test_iter_A = LoadSig(dataset)
    for i, sig in enumerate(test_iter_A):
        sig = sig.cuda()
        out = model(sig)  # [sigPred, vLossCoeff, g_last, hl]
    plot_box(out[1], batchidx=0, title='Nomral signal of model (before threshold)', mode='WPT', m=m,fontsize=fontsize)
    plot_box(out[-1], batchidx=0, title='Nomral signal of model (after threshold)',m=m,fontsize=fontsize)
    plt.figure()
    plt.plot(np.arange(dataset.shape[2]), dataset[indPlot, 0, :, 0], color, linewidth=linewidth)
    plt.plot(np.arange(dataset.shape[2]), out[0][indPlot, 0, :, 0].detach().cpu().numpy(), color2,
             linewidth=linewidth)
    plt.legend(['Original Signal ', 'Reconstructed signal'],fontsize=24)
    plt.tick_params(labelsize=fontsize)
    plt.xlabel('Time point', fontsize=fontsize)
    plt.ylabel('Amplitude', fontsize=fontsize)
    plt.ylim((-2, 2))
    plt.tight_layout()
    plt.show()
    reconstruction_error = np.mean((dataset[indPlot, 0, :, 0] - out[0][indPlot, 0, :, 0].detach().cpu().numpy()) ** 2)
    print(title)
    print('重构误差 {}'.format(reconstruction_error))

# def kernel_visual(kernel):
#     t=1
if __name__ == '__main__':
    seed_torch(666)
    #共分2个通道，一个低频10Hz，一个高频240Hz，采频500Hz
    fs = 500
    n = np.arange(1, 4 * fs)# 2s时间，1s训练，1s测试
    t = n / fs
    signal_1 = np.sin(10 * 2 * np.pi * t)#正常信号
    signal_2 = 0.5*np.sin(10* 2 * np.pi * t)#低幅值异常
    signal_3 = 2.0*np.sin(10* 2 * np.pi * t)#高幅值异常
    signal_4 = np.sin(240 * 2 * np.pi * t)#高周期异常

    lTrain = 1000

    signalT1 = signal_1[np.newaxis, np.newaxis, :, np.newaxis]
    signalT2 = signal_2[np.newaxis, np.newaxis, :, np.newaxis]
    signalT3 = signal_3[np.newaxis, np.newaxis, :, np.newaxis]
    signalT4 = signal_4[np.newaxis, np.newaxis, :, np.newaxis]

    signal = signalT1[:, :, :lTrain, :]
    signal_A = signalT1[:, :, lTrain:, :]#正常信号

    signal_test_B1 = signalT2[:, :, :lTrain, :]#低幅值异常
    signal_B1 = signalT2[:, :, lTrain:, :]

    signal_test_B2 = signalT3[:, :, :lTrain, :]#高幅值异常
    signal_B2= signalT3[:, :, lTrain:, :]

    signal_test_B3 = signalT4[:, :, :lTrain, :]#高周期异常
    signal_B3 = signalT4[:, :, lTrain:, :]

    trainHT = True

    # Weight for sparsity loss versus residual?

    lossCoeff_setting = True
    coeffLoss = CoeffLoss(lossCoeff_setting)
    recLoss = RecLoss()

    epochs = 1000
    m = [2]
    k = [4]
    level = len(m)
    mode = 'WPT'
    visualization_kernel = True
    device = 'cuda'
    # Kernel_I = Kernel_selector(band=3)
    model = WADNet(kernelInit=2, m=m, k=k, level=level, kernelsConstraint='KIPL', mode=mode, device=device,
                 coefffilter=5.0, realconv=True, initHT=0.01, initHT1=3, kernTrainable=True, threshold=True,t=100.0)
    model.cuda()

    opt = torch.optim.Adam(params=model.parameters(), lr=0.03, betas=(0.9, 0.999), eps=1e-07)
    lr_scheduler = lr_scheduler_choose(lr_scheduler_way='step', optimizer=opt, steps='800,1300,1400', gamma=0.1)

    kernels = model.MBAND
    kernellen = m[0] * k[0]

    # threshold_list = model.thre  # 每一层有一个
    # for i in range(m[0]-1):
    #     print('Before'+'*' * 6 + 'threshold in first layer of band!!'+str(i) + '*' * 6)
    #     print(threshold_list[i].left)
    #     print(threshold_list[i].right)
    print('threshold before optimizaiton')
    threshold_number = model.thre.l.size(0)
    print('左侧内侧')
    print(model.thre.l.view(threshold_number))
    print('左侧带通范围')
    print(model.thre.l1.view(threshold_number))
    print('右侧内侧')
    print(model.thre.r.view(threshold_number))
    print('右侧带通范围')
    print(model.thre.r1.view(threshold_number))


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
            # sigPred,vLossCoeff,g_last,hl = model(sig)
            # r=recLoss(sig,sigPred)
            # c=coeffLoss(sigPred)
            # loss=r+c
            # loss.backward(retain_graph=True)
            sig = sig.cuda()
            # sigPred, coeff, loss2, _ = model(sig)
            sigPred, coeff, loss2, hard_threshold_max, visualization_coeff_after_threshold = model(sig)
            loss1 = coeffLoss(coeff) # 稀疏项
            r = recLoss(sig, sigPred) #重构loss
            # loss = r #仅重构
            # loss = r + 10.0 * loss2 #重构 + 滤波器约束
            # loss = r +  loss1 #重构 + 稀疏
            # loss = r +loss1 +10.0*loss2 #重构+稀疏+滤波器约束
            # loss = r - 0.1*sigmoid(hard_threshold_max)# 重构 + 阈值带通区间最小化
            # loss = r + 1.0*loss1 - 1.0*sigmoid(hard_threshold_max)# 重构 + 稀疏 + 阈值带通区间最小化
            # loss = r + 1.0*loss1 - 1.0*sigmoid(hard_threshold_max) + 10.0 * loss2# 重构 + 稀疏 + 阈值带通区间最小化 + 滤波器约束

            # loss = r + 0.1*torch.log(sigmoid(hard_threshold_max)+1)# 重构 + 阈值带通区间最小化
            # loss = r + 0.3 * loss1 + 0.3 * loss2
            # loss = r + 0.3 * loss1 + 0.3 * loss2- 0.1*torch.log(hard_threshold_max+1) # 重构+稀疏+正交+阈值最大化
            # loss = 2.0*r + 1.0*loss1 - 0.5*hard_threshold_max + 1.0 * loss2# 重构 + 稀疏 + 阈值带通区间最小化 + 滤波器约束
            loss = 1.0*r + 0.5*loss1 - 0.25*hard_threshold_max + 0.5 * loss2# 重构 + 稀疏 + 阈值带通区间最小化 + 滤波器约束

            # loss =  loss1
            # loss =  0.01 * loss2
            loss.backward()
            opt.step()
            running_loss += r.item()
            lr_scheduler.step()
        print('[epoch:%d/%d] loss: %.3f'
              % (epoch + 1, epochs, running_loss / len(train_iter)))

    # coefficient_visul(visualization_coeff_after_threshold, mode=mode, m=m,title='train')  # ,prefined_sort=[1,2,0,3]

    indPlot = 0
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

    plot_experiment(signal_A,title='Normal signal',model=model,indPlot=indPlot,m=m,color='b',color2='g')
    plot_experiment(signal_B1,title='Low amplitude signal',model=model,indPlot=indPlot,m=m,color='r',color2='g')
    plot_experiment(signal_B2,title='High amplitude signal',model=model,indPlot=indPlot,m=m,color='r',color2='g')
    plot_experiment(signal_B3,title='High period signal',model=model,indPlot=indPlot,m=m,color='r',color2='g')


