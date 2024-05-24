# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd


import matplotlib
# del matplotlib.font_manager.weight_dict['roman']
# matplotlib.font_manager._rebuild()
import matplotlib.pyplot as plt
plt.rc('font',family='Times New Roman')
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

indPlot = 0



fs = 500
n = np.arange(1, 0.5 * fs)# 0.5时间
t = n / fs
signal_1 = np.sin(20 * 2 * np.pi * t)
signal_2 = np.sin(50* 2 * np.pi * t)
signal_3 = 2*np.sin(20* 2 * np.pi * t)

lTrain = 249


# ax.title.set_text('Signal A')
plt.rcParams['figure.figsize']=(8,5)


fontsize = 32

plt.figure()
plt.plot(t,signal_1,linewidth=4,color='b')#'#ff7f0e' '#1f77b4'
plt.tick_params(labelsize=fontsize)
plt.ylim((-2, 2))
plt.show()

plt.figure()
plt.plot(t,signal_2,linewidth=4,color='r')#'#ff7f0e' '#1f77b4'
plt.tick_params(labelsize=fontsize)
plt.ylim((-2, 2))
plt.show()

plt.figure()
plt.plot(t,signal_3,linewidth=4,color='r')#'#ff7f0e' '#1f77b4'
plt.tick_params(labelsize=fontsize)
plt.ylim((-2, 2))
plt.show()


