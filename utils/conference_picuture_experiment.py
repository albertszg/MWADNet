# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataset import LoadSig
import torch
import torch.nn as nn
from utils.torch_random_seed import seed_torch
from scipy import interpolate
import seaborn as sns
from paper_model.FMBWN import FMWN
from utils.utils_ae import lr_scheduler_choose
from matplotlib.colors import ListedColormap

'''
(default: cycler('color', ['#1f77b4', '#ff7f0e', '#2ca02c', 
'#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', 
'#bcbd22', '#17becf'])).

'''

def fft_fignal_plot(signal, fs, window, plot=True,title='spectrum',color='b'):
    sampling_rate = fs
    fft_size = window
    # t = np.arange(0, fft_size/sampling_rate, 1.0 / sampling_rate)
    fontsize = 32
    linewidth = 5
    t = np.arange(0, fft_size, 1)
    mean = signal.mean()
    xs = signal[:fft_size] - mean
    xf = np.fft.rfft(xs) / fft_size
    freqs = np.linspace(0.0, sampling_rate / 2.0, int(fft_size / 2 + 1))
    xfp = np.abs(xf)
    # xfp = 20 * np.log10(np.clip(np.abs(xf), 1e-20, 1e100))
    if plot:
        plt.figure(figsize=(12, 6))
        plt.subplot(211)
        plt.plot(t[:fft_size], xs, color=color,linewidth=linewidth)
        plt.xlabel("Time/s", fontsize= fontsize)
        plt.ylabel('Amplitude', fontsize= fontsize)
        plt.ylim((-2, 2))
        # plt.title("signal")
        plt.tick_params(labelsize= fontsize)

        plt.subplot(212)
        plt.plot(freqs, xfp, color=color,linewidth=linewidth)
        plt.xlabel("Frequency/Hz", fontsize= fontsize)
        # 字体FangSong
        plt.ylabel('Amplitude', fontsize= fontsize)
        plt.ylim((0, 1))
        plt.tick_params(labelsize= fontsize)
        plt.subplots_adjust(hspace=0.4)
        # plt.title(title)
        '''subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
        有六个可选参数来控制子图布局。值均为0~1之间。其中left、bottom、right、top围成的区域就是子图的区域。
        wspace、hspace分别表示子图之间左右、上下的间距。实际的默认值由matplotlibrc文件控制的。
        '''
        plt.tight_layout()
        plt.show()



indPlot = 0
window = 999

fs = 500
n = np.arange(1, 2 * fs)# 0.5时间
t = n / fs
signal_1 = np.sin(10 * 2 * np.pi * t)
signal_2 = 0.5*np.sin(10* 2 * np.pi * t)
signal_3 = 2.0*np.sin(10* 2 * np.pi * t)
signal_4 = np.sin(240* 2 * np.pi * t)

fft_fignal_plot(signal_1,fs,window ,plot=True,title='signal1',color='b')
fft_fignal_plot(signal_2,fs,window ,plot=True,title='signal1',color='r')
fft_fignal_plot(signal_3,fs,window ,plot=True,title='signal1',color='r')
fft_fignal_plot(signal_4,fs,window ,plot=True,title='signal1',color='r')

#
# # ax.title.set_text('Signal A')
# plt.figure()
# plt.plot(t,signal_1,linewidth=4,color='b')#'#ff7f0e' '#1f77b4'
# plt.tick_params(labelsize=18)
# plt.ylim((-2, 2))
# plt.show()
#
# plt.figure()
# plt.plot(t,signal_2,linewidth=4,color='r')#'#ff7f0e' '#1f77b4'
# plt.tick_params(labelsize=18)
# plt.ylim((-2, 2))
# plt.show()
#
# plt.figure()
# plt.plot(t,signal_3,linewidth=4,color='r')#'#ff7f0e' '#1f77b4'
# plt.tick_params(labelsize=18)
# plt.ylim((-2, 2))
# plt.show()


