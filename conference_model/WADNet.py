# -*- coding: utf-8 -*-
#提供网络的结构，组合utils中相关的基本component
import numpy as np
import torch
import torch.nn as nn
import utils.mband_Layers as impLay
from utils.kernel_selector import Kernel_selector

class CoeffLoss(nn.Module):
    def __init__(self,loss=True,mode = 'WPT'):
        super(CoeffLoss, self).__init__()
        self.loss = loss
        self.mode = mode
    def forward(self,all_coefficients):
       if self.loss:
            if self.mode == 'WPT':
                node_number = len(all_coefficients)  # 分解出的节点小波系数
                for i in range(node_number):
                    if i == 0:
                        vLossCoeff = torch.mean(torch.abs(all_coefficients[i]))
                    else:
                        vLossCoeff = vLossCoeff + torch.mean(torch.abs(all_coefficients[i]))
                # vLossCoeff = torch.div(vLossCoeff, node_number)
                return torch.div(vLossCoeff, node_number)
            else:
                list_len = len(all_coefficients)
                node_number = 0
                #原版程序
                for i in range(list_len):
                    lev_coefficients = all_coefficients[i]
                    lev_len = len(lev_coefficients)
                    for j in range(lev_len):
                        if i==0 and j==0:
                            vLossCoeff = torch.mean(torch.abs(lev_coefficients[j]))
                        else:
                            vLossCoeff = vLossCoeff + torch.mean(torch.abs(lev_coefficients[j]))
                        node_number +=1
                return torch.div(vLossCoeff, node_number)

                # for i in range(list_len):
                #     lev_coefficients = all_coefficients[i]
                #     lev_len = len(lev_coefficients)
                #     for j in range(lev_len):
                #         if i==0 and j==0:
                #             vLossCoeff = torch.sum(torch.abs(lev_coefficients[j]))
                #         else:
                #             vLossCoeff = vLossCoeff + torch.sum(torch.abs(lev_coefficients[j]))
                #         # node_number += 1
                #         node_number += lev_coefficients[j].numel()

                return torch.div(vLossCoeff, node_number)
       else:
           # return all_coefficients
           return 0
class WADNet(nn.Module):
    """
        Function that generates a torch FMWN network

        Parameters
        ----------
        inputSize : INT, optional
            Length of the time series. Network is more efficient if set.
            Can be set to None to allow various input size time series.
            The default is None.

        kernelInit: LIST consist of M numpy array
                    the kernelInit is the kernel
                    only used for CF constrains

        M: LIST consist of INT, optional
            Initialisation of the kernel.
            the length of FIR filter = M[i]*K[i]
            The default is [2]. one layer decomposition

        Regularity (k): LIST consist of INT, optional
            the length of FIR filter = M[i]*K[i]
            The default is [3]. one layer decomposition

        Mode : String, optional
            'WPT' wavelet packet transform 小波包
            'DWT': Only scaling filter is used to decompose signal
            'Cus': Specify the coefficients you want to decompose. Not implemented

        kernTrainable : BOOL, optional
            Whether the kernels are trainable. Set to FALSE to compare to traditional wavelet decomposition.
            The default is True.

        kernelsConstraint : STRING, optional
                'KI: 核初始化; CF：所有层使用一个核; '
                'KIPL: 初始化的同时每层不一样；PL: perlayer每层独立的核 '

        initHT : FLOAT, optional
            Value to initialise the Hard-thresholding coefficient.
            The default is 1.0.
        trainHT : BOOL, optional
            Whether the hard-thresholding coefficient is trainable or not.
            Set FALSE to compare to traditional wavelet decomposition.
            The default is True.
        t: temperature for hard thresholding default is 20

        Returns
        -------
        model1: a torch neural network with outputs the :
        1.reconstructed signals
        2.the loss on the wavelet coefficients: L1 loss
        3.the loss on filter
        4.wavelet coefficients
        """
    def __init__(self, kernelInit=None, m=[3], k=[2],level=None, mode='WPT',kernTrainable=True,  kernelsConstraint='CF',
                 coefffilter=1.0,device='cuda',realconv=True,t=100.0,initHT=1.0, initHT1=5.0,trainHT=True,threshold=True,seqlen=1000):
        super(WADNet, self).__init__()

        self.m = m
        self.k = k
        assert len(self.m)==len(self.k)
        self.level = len(m)
        self.mode=mode
        self.FilterCoeff = coefffilter

        if kernelsConstraint == 'CF':
            '''
            此模式下所有层 带数和正则数 一样，设置为m和k的第一项!!
            一组滤波器生成DWT所有滤波器，正交滤波器组
            限定长度可学习时就是网络
            '''
            ''' kernel setting'''
            if isinstance(level, int):
                self.level = level
            self.multilayer_constraints=False #多层滤波器约束，只有单层
            self.lossFilters = 'ON'

            De = [impLay.Kernel(bandM=self.m[0], regularity=self.k[0], trainKern=kernTrainable,device=device)() for _ in
                  range(self.m[0])]
            ''' mband function'''
            mband = [impLay.Mband(De=De, Re=De, M=self.m[0], k=self.k[0],realconv=realconv, device=device) for _ in range(self.level)]
            ''' list to torch module'''
            self.MBAND = nn.ModuleList(mband)


        elif kernelsConstraint == 'KI':
            '''
            kernelInit
            利用滤波器系数建立M-般的分解，不可学习时就是M-band分解，
            可学习就是利用已有的小波系数初始化 + CF约束学习模式？：合成滤波器由分解滤波器确定，每一层都是一个，
            '''
            self.multilayer_constraints = False
            self.lossFilters = 'ON'
            if isinstance(level, int):
                self.level = level
            kernelInit = Kernel_selector(kernelInit)
            De = [impLay.Kernel(bandM=kernelInit[band], trainKern=kernTrainable,device=device)() for band in range(self.m[0])]
            mband = [impLay.Mband(De=De, Re=De, M=self.m[0], k=self.k[0],realconv=realconv, device=device) for _ in range(self.level)]
            self.MBAND = nn.ModuleList(mband)

        elif kernelsConstraint == 'KIPL':
            '''
            kernelInit + PerLayer 
            利用滤波器系数建立M-般的分解，不可学习时就是M-band分解，
            可学习就是利用已有的小波系数初始化 + CF约束学习模式？：合成滤波器由分解滤波器确定，每一层都是一个，
            '''
            self.multilayer_constraints = True
            self.lossFilters = 'ON'
            if isinstance(level, int):
                self.level = level
            # De = [impLay.Kernel(bandM=kernelInit[band], trainKern=kernTrainable,device=device)() for band in range(self.m[0])]
            De = []
            for lev in range(self.level):# 层
                if self.m[lev]==2:
                    kernel_temp = Kernel_selector(1)
                else:
                    kernel_temp = Kernel_selector(self.m[lev])
                De.append([impLay.Kernel(bandM=kernel_temp[band], trainKern=kernTrainable, device=device)() for band in
                  range(self.m[lev])]) # 每层的kernel
            # mband = [impLay.Mband(De=De, Re=De, M=self.m[0], k=self.k[0],realconv=realconv, device=device) for _ in range(self.level)]
            mband = [impLay.Mband(De=De[lev], Re=De[lev], M=self.m[lev], k=self.k[lev], realconv=realconv, device=device) for lev in range(self.level)]
            self.MBAND = nn.ModuleList(mband)

        elif kernelsConstraint == 'PL':
            '''
            PerLayer
            满足M-band小波条件
            每一层mband不一样: m不一样 k不一样
            '''
            self.multilayer_constraints = True
            self.lossFilters = 'ON'

            De=[]
            for lev in range(self.level):# 层
                De.append([impLay.Kernel(bandM=self.m[lev], regularity=self.k[lev], trainKern=kernTrainable,device=device)() for _ in
                  range(self.m[lev])]) # 每层的kernel
            mband = [impLay.Mband(De=De[lev], Re=De[lev], M=self.m[lev], k=self.k[lev],realconv=realconv, device=device) for lev in range(self.level)]
            self.MBAND = nn.ModuleList(mband)


        elif kernelsConstraint == 'Free':
            '''
            所有高低通以及重构滤波器都有自己的核，及无约束
            '''
            self.lossFilters = 'OFF'
            De = []
            Re = []
            for lev in range(self.level):
                De.append([impLay.Kernel(bandM=self.m[lev], regularity=self.k[lev], trainKern=kernTrainable,device=device)() for _ in range(self.m[lev])])
                Re.append([impLay.Kernel(bandM=self.m[lev], regularity=self.k[lev], trainKern=kernTrainable,device=device)() for _ in range(self.m[lev])])
            mband = [impLay.Mband(De=De[lev], Re=Re[lev], M=self.m[lev], k=self.k[lev],realconv=realconv, device=device) for lev in range(self.level)]
            self.MBAND = nn.ModuleList(mband)


        elif kernelsConstraint == 'PerLayer_wavelet':
            '''
            每一层mband不一样, 但低通滤波器一样？
            '''
            raise Exception('kernelsConstraint not implemented')
        else:
            raise Exception('kernelsConstraint not found!')

        '''
        设置阈值滤波激活函数
        impLay.WaveletThreshold 原始我设置的阈值
        impLay.wavelet_threshold_normal_min_max(x,t,left,right,left1,right1,norm=False)
        '''

        # print(self.MBAND[0].M)
        thre = []
        if mode == 'DWT':
            thre = [impLay.WaveletThreshold(init=initHT, trainBias=trainHT, t=t, m=i-1,threshold=threshold)  for i in self.m]
            self.thre = nn.ModuleList(thre)
            self.thre_a = impLay.WaveletThreshold(init=initHT, trainBias=trainHT, t=t, m=1,
                                                  threshold=threshold)  # DWT模式时最后一次分解后对低频进行处理的阈值模块
        elif mode == 'WPTplus':
            # 方案2，在每一层都使用一次阈值处理
            thre = []
            tmp = 1
            self.WPTNodeNumber = []
            for m_band in self.m:# 计算最后一层的节点数
                tmp = tmp * m_band
                self.WPTNodeNumber.append(tmp)
                thre.append(impLay.WaveletThreshold_plus(init=initHT, init1=initHT1,trainBias=trainHT, t=t, m=tmp,threshold=threshold))
            self.thre = nn.ModuleList(thre)
        else:
            #方案1，仅在最后一层使用阈值
            tmp = 1
            for m_band in self.m:# 计算最后一层的节点数
                tmp = tmp * m_band
            self.WPTNodeNumber = tmp
            self.thre = impLay.WaveletThreshold_plus(init=initHT, init1=initHT1,trainBias=trainHT, t=t, m=self.WPTNodeNumber,threshold=threshold)# 一共使用m*4个可学习参数
            # self.thre = nn.ModuleList(thre)

        # self.dropout = nn.Dropout(0.3)
        # self.dropout1d = nn.Dropout1d(0.3)#对通道进行整体置零

    def forward(self, inputSig,noDHT=True,ht=None):#inputSig size: batch*channel*length*hight   hight = 1
        assert inputSig.size()[-1]==1
        device=inputSig.device
        if self.mode=='DWT':
            '''分解第一个系数'''
            low_passed_signal=inputSig
            coefficients=[]#由列表组成的系数，用于网络内部运算时候使用，阈值处理后的
            all_coefficients=[] # 用于计算稀疏约束的系数集合
            visual_coefficients=[] #用于可视化的系数集合
            for lev in range(self.level):
                # NEW 可视化阈值处理‘后’的信号，稀疏约束使用阈值处理‘前’的系数
                c_tmp = self.MBAND[lev](low_passed_signal)
                all_coefficients.append(c_tmp[1:])#计算原本系数的稀疏性
                c_tmp[1:] = self.thre[lev](c_tmp[1:], NodefineT=noDHT, HT=ht) # 对高频部分进行阈值操作
                coefficients.append(c_tmp[1:])#保存阈值处理后的系数
                visual_coefficients.append(c_tmp[1:])#可视化是处理后的系数
                low_passed_signal = c_tmp[0]#最后一层低通的信号
                #back up
                #初始版本 可视化阈值处理前的信号，稀疏约束使用阈值处理‘后’的系数
                # c_tmp = self.MBAND[lev](low_passed_signal)
                # visual_coefficients.append(c_tmp[1:])  # 可视化未被阈值处理前的信号时频图
                # coefficients.append(self.thre[lev](c_tmp[1:], NodefineT=noDHT,HT=ht))#
                # all_coefficients.append(c_tmp[1:])#原来的，阈值并没有加入系数统计中，也就没有参与进稀疏的度量里面去
                # low_passed_signal=c_tmp[0]#低通逼近系数

                #NEW 可视化阈值处理‘前’的信号，稀疏约束使用阈值处理‘后’的系数
                # c_tmp = self.MBAND[lev](low_passed_signal)
                # visual_coefficients.append(c_tmp[1:])  # 可视化未被阈值处理前的信号时频图
                # c_tmp[1:] = self.thre[lev](c_tmp[1:], NodefineT=noDHT,HT=ht)#处理后的系数
                # coefficients.append(c_tmp[1:] )#添加到系数集合，后面重构使用
                # all_coefficients.append(c_tmp[1:])#稀疏度测量
                # low_passed_signal=c_tmp[0]#低通逼近系数
            #不对低频处理
            # all_coefficients.append([low_passed_signal]) # 稀疏无阈值处理的系数
            # visual_coefficients.append(low_passed_signal)  # 可视化无阈值处理的系数
            # re_signal = low_passed_signal #重构
            #对低频也处理
            # visual_coefficients.append([low_passed_signal])
            all_coefficients.append([low_passed_signal]) # 稀疏无阈值处理的系数
            low_passed_signal =self.thre_a([low_passed_signal], NodefineT=noDHT,HT=ht)
            # all_coefficients.append(low_passed_signal)# 稀疏约束处理后的系数
            visual_coefficients.append(low_passed_signal)# 可视化阈值处理后的系数
            re_signal = low_passed_signal[0] #重构
            low_pass_t=True # 表征是否对最后一层的低通部分阈值处理，处理的话后面阈值优化也需要开启对应优化。

            #开始重构信号
            for lev in range(self.level-1,-1,-1):
                temp=[]
                # #未加入dropout版
                temp.append(re_signal)
                temp.extend(coefficients[lev])
                re_signal = self.MBAND[lev](input=temp, decomposition=0)
            ### 计算硬阈值和，使得频带通过的内容尽可能的窄，即使得l和r尽可能大，而l1和r1尽可能小
            ### 新的定义下，使得l和r尽可能大, l1和r1尽可能小
            hard_number = 0
            for lev in range(self.level):
                if lev == 0:
                    hard_threshold = torch.sum(self.thre[lev].l1 - self.thre[lev].l)
                    hard_threshold += torch.sum(self.thre[lev].r1 - self.thre[lev].r)
                else:
                    hard_threshold += torch.sum(self.thre[lev].l1 - self.thre[lev].l)
                    hard_threshold += torch.sum(self.thre[lev].r1 - self.thre[lev].r)
                hard_number += 1.0
            if low_pass_t:
                hard_threshold += torch.sum(self.thre_a.left)
                hard_threshold += torch.sum(self.thre_a.right)
                hard_number += 1.0
            hard_threshold = hard_threshold / hard_number
        elif self.mode=='WPT':
            '''分解所有系数
            需要考虑小波系数重排的问题
            奇数不变，偶数变
            方案1：
            仅最后一层节点加入阈值处理
            方案2：
            fink 论文里面是每次分解都加入阈值模块
            WPT的滤波器每层的滤波器在不同带复用，所以就每层一套M带滤波器，建立一个约束即可，跟DWT一致
            '''
            # low_passed_signal = inputSig
            # coefficients = []  # 由列表组成的系数，用于网络内部运算时候使用，阈值处理后的
            all_coefficients = []  # 用于计算稀疏约束的系数集合
            visual_coefficients = []  # 用于可视化的系数集合

            low_passed_signal=[inputSig]
            for lev in range(self.level):
                tmp=[]
                for Mcoefficients in low_passed_signal:
                    filtered_signal = self.MBAND[lev](Mcoefficients) #对所有的带(一个列表)都进行m带的分解
                    #若每层都需要阈值处理，则在此处加入阈值处理操作
                    tmp.extend(filtered_signal) #分解后的信号
                    # if lev < self.level-1:
                    all_coefficients.extend(filtered_signal) #用于计算稀疏约束的系数集合
                low_passed_signal = tmp
            low_passed_signal = self.thre(low_passed_signal, NodefineT=noDHT,HT=ht)#对最后一层的分解结果进行阈值处理

            # all_coefficients.extend(low_passed_signal)# 将阈值处理项纳入稀疏优化
            re_signal = low_passed_signal# node 节点个数组成的 tensor
            visual_coefficients = low_passed_signal #使用阈值处理后的系数可视化
            for lev in range(self.level - 1, -1, -1):
                temp = []
                m_temp=self.MBAND[lev].M
                len_temp = len(re_signal)
                for i in range(0,len_temp,m_temp):
                    temp.append(self.MBAND[lev](input=re_signal[i:i+m_temp], decomposition=0))
                re_signal=temp
            re_signal=re_signal[0]

            ### 计算硬阈值和，使得频带通过的内容尽可能的窄，即使得l和r尽可能大，而l1和r1尽可能小
            ### 新的定义下，使得l和r尽可能大, l1和r1尽可能小,即 l-|l1|尽可能大,其为正数
            # hard_number = 0
            # for lev in range(self.level):
            #     if lev == 0:
            #         hard_threshold = torch.sum(self.thre.l-torch.abs(self.thre.l1))
            #         hard_threshold += torch.sum(self.thre.r-torch.abs(self.thre.r1))
            #     else:
            #         hard_threshold += torch.sum(self.thre.l-torch.abs(self.thre.l1))
            #         hard_threshold += torch.sum(self.thre.r-torch.abs(self.thre.r1))
            #     hard_number += 1.0
            # # hard_threshold = hard_threshold / hard_number
            hard_threshold_init = torch.sum(self.thre.l) + torch.sum(self.thre.r)
            hard_threshold_bandwidth = torch.sum(torch.square(self.thre.l1)) + torch.sum(torch.square(self.thre.r1))
            hard_threshold = 5.0*torch.sigmoid(hard_threshold_init/ self.WPTNodeNumber) - hard_threshold_bandwidth/ self.WPTNodeNumber
            # hard_threshold = hard_threshold_init/ self.WPTNodeNumber - hard_threshold_bandwidth/ self.WPTNodeNumber

        else:
            raise Exception('Decomposition mode not found!')

        #计算滤波器约束项
        if self.lossFilters == 'ON':
            #约束平方和，好优化 MSE
            if self.multilayer_constraints:
                for lev in range(self.level):
                    filter_list = self.MBAND[lev].filter()
                    stacked_filter = torch.stack(filter_list).squeeze()
                    m = self.m[lev]
                    k = self.k[lev]
                    ''' the low-pass and high-pass filter condition '''
                    for i in range(0, m):
                        if i == 0:
                            filter_condition = (torch.sum(filter_list[i]) - torch.sqrt(
                                torch.tensor(m, device=device, dtype=float))) ** 2
                        else:
                            filter_condition = filter_condition + (torch.sum(filter_list[i])) ** 2
                    ''' the orthonormal_condition '''
                    eye_target = torch.eye(k * m, device=device)  # 单位阵
                    P = torch.zeros([k * m, k * m + (k - 1) * m], dtype=torch.float, device=device)
                    Q = torch.zeros([k * m, k * m + (k - 1) * m], dtype=torch.float, device=device)

                    for i in range(k):  # 块的行数
                        P[i * m:m * (i + 1), i * m:(i * m + m * k)] = stacked_filter[:, :]

                    for i in range(k):  # 块的行数,第i行块
                        for j in range(k):  # 第j列块
                            Q[i * m:(i + 1) * m, (i * m + j * m):(i * m + (j + 1) * m)] = stacked_filter[:,
                                                                                          j * m:(j + 1) * m].T

                    orthonormal_condition = torch.sum(torch.square(torch.mm(P, P.T) - eye_target)) + torch.sum(
                        torch.square(torch.mm(Q, Q.T) - eye_target))  # 元素平方
                    if lev ==0:
                        vLossFilters = self.FilterCoeff * filter_condition + orthonormal_condition
                    else:
                        vLossFilters = self.FilterCoeff * filter_condition + orthonormal_condition + vLossFilters
            else:
                filter_list = self.MBAND[0].filter()
                stacked_filter = torch.stack(filter_list).squeeze()
                m=self.m[0]
                k=self.k[0]
                ''' the low-pass and high-pass filter condition '''
                for i in range(0,m):
                    if i == 0:
                        filter_condition = (torch.sum(filter_list[i]) - torch.sqrt(torch.tensor(m,device=device,dtype=float)))**2
                    else:
                        filter_condition = filter_condition + (torch.sum(filter_list[i]))**2
                ''' the orthonormal_condition '''
                eye_target = torch.eye(k*m,device=device) #单位阵
                P = torch.zeros([k*m,k*m+(k-1)*m],dtype=torch.float,device=device)
                Q = torch.zeros([k*m,k*m+(k-1)*m],dtype=torch.float,device=device)

                for i in range(k):#块的行数
                    P[i*m:m*(i+1),i*m:(i*m+m*k)]=stacked_filter[:,:]

                for i in range(k):#块的行数,第i行块
                    for j in range(k):#第j列块
                        Q[i * m:(i + 1) * m, (i * m + j * m):(i * m + (j + 1) * m)] = stacked_filter[:,j*m:(j+1)*m].T

                orthonormal_condition = torch.sum(torch.square(torch.mm(P,P.T)-eye_target)) + torch.sum(torch.square(torch.mm(Q,Q.T)-eye_target))#元素平方

                vLossFilters = self.FilterCoeff*filter_condition + orthonormal_condition

        elif self.lossFilters == 'OFF':
            vLossFilters = torch.tensor(0.0,device=device)
        else:
            raise ValueError(
                'Could not understand value in \'lossCoeff\'. It should be either \'ON\' or \'OFF\'')

        # 需要改成返回小波系数，在主程序计算，并用于画图，filter loss可以在这里计算，special 版本需求在改成返回filter
        return re_signal,all_coefficients,vLossFilters,hard_threshold,visual_coefficients # 重构的信号， 用于计算稀疏约束的系数，滤波器的约束，硬阈值约束，用于可视化的系数

if __name__ == '__main__':
    # signal=torch.randn(1,1,64,1,dtype=torch.float32)
    signal = torch.Tensor(range(256)) - 128.0
    signallen = signal.size(0)
    signal = signal.view(1, 1, signallen, 1)
    kernelInit=2 #带数量
    model=WADNet(kernelInit=kernelInit,m=[2,2],k=[2,2],level=2,kernelsConstraint='KI',mode='WPT')
    # for name,param in model.named_parameters():
    #     print(name,param)
    model.to('cuda')
    signal = signal.to('cuda')
    loss1, loss2, loss3, loss4, loss5=model(signal)
    a=2
    print(a)
    # print(out[0].reshape(1,-1))
    # out2=model(signal2)
    # graph=make_dot(out[0],params=dict(model.named_parameters()))
    # graph.view()
    # print(out2[0].shape,)
    # print(out2[1],)
    # print(out2[2].shape, )

