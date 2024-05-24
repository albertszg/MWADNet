#!/usr/bin/env python
# -*- coding: utf-8 -*-
import warnings
import os
from time import time
from copy import deepcopy
import scipy.io
import torch
from torch import optim
import torch.nn as nn
from torch.nn import MSELoss
import numpy as np
import logging
import matplotlib.pyplot as plt
from utils.utils_ae import metrics_calculate_mband, metrics_calculate, lr_scheduler_choose
from utils.t_sne import T_SNE
from paper_model.despwan import DeSpaWN

'''
FMWN(kernelInit=None, m=[3], k=[2],level=None, mode='DWT',kernTrainable=True,  kernelsConstraint='CF',
                 coefffilter=1.0,device='cuda',realconv=True,t=20.0,initHT=1.0, trainHT=True,threshold=True)
Kernel_selector:1,2,3,4,5
coefficient_visul_pywt(all_coeff,mode='DWT')
coefficient_visul(all_coeff,mode='DWT', m=None,sort='freq',prefined_sort=None)
'''
def concate_coeff(z):#z: list=[[c1,c2],[c3,c4],[]]
        for i in range(len(z)):
            for j in range(len(z[i])):
                if i==0 and j==0:
                    temp=z[i][j].detach()
                else:
                    temp=torch.cat((temp,z[i][j].detach()),dim=2)
        return temp

class SparseDWT_AE(object):
    def __init__(self, data_loader, savedir, args):
        # Consider the gpu or cpu condition
        self.args = args
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            logging.info('using {} gpus'.format(self.device_count))
            assert args.batch_size % self.device_count == 0, "batch size should be divided by device count"
        else:
            warnings.warn("gpu is not available")
            self.device = torch.device("cpu")
            self.device_count = 1
            logging.info('using {} cpu'.format(self.device_count))
        #inp_dim=data_loader['nc'], z_dim=args.z_dim,seqlen=data_loader['len']

        # kernel_init = Kernel_selector(args.KernelInit)
        if self.args.DWTInit==1:
            kernelInit = np.array([-0.010597401785069032, 0.0328830116668852, 0.030841381835560764, -0.18703481171909309,
                               -0.027983769416859854, 0.6308807679298589, 0.7148465705529157, 0.2303778133088965])
        else:
            kernelInit = self.args.DWTInit
        ae = DeSpaWN(kernelInit =kernelInit, kernTrainable = self.args.DWTkernTrainable, level = self.args.DWTlevel,
                     lossCoeff = 'l1', kernelsConstraint = self.args.DWTmode,initHT = self.args.DWTinitHT, trainHT = self.args.DWTtrainHT)


        logging.info(ae)

        self.savedir = savedir
        self.polluted_ratio = args.snr# SNR
        self.data_prefix = args.key_information
        self.filterlosscoeff=args.filterlosscoeff
        self.sparsitylosscoeff=args.DWTsparsitylosscoeff

        self.lr = args.lr
        self.epoch = args.epoch

        self.early_stop = args.early_stop
        self.early_stop_tol = args.early_stop_tol

        self.plot_confusion_matrix = args.plot_confusion_matrix

        self.ae = ae.to(self.device)
        if self.device_count > 1:
            self.ae = torch.nn.DataParallel(self.ae)
        self.data_loader = data_loader  # 加载数据器：train, val, test, nc

        if args.loss_function=='MSE':
            self.mse = MSELoss()  # 训练ae
        elif args.loss_function=='MAE':
            self.mse = nn.L1Loss()  # 训练ae
        else:
            raise Exception('Other loss funciton is not supported')
        # self.coff_sparsity = CoeffLoss(loss=args.coeff_sparsity,mode = args.mband_mode)


        if args.opt == 'adam':
            self.ae_optimizer = optim.Adam(self.ae.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.opt == 'sgd':
            self.ae_optimizer = optim.SGD(self.ae.parameters(), lr=args.lr, momentum=args.momentum,
                                          weight_decay=args.weight_decay)
        else:
            raise Exception("optimizer not implement")

        self.lr_scheduler = lr_scheduler_choose(lr_scheduler_way=args.lr_scheduler, optimizer=self.ae_optimizer,
                                                steps=args.scheduler_step_size, gamma=args.scheduler_gamma)

        self.cur_step = 0
        self.cur_epoch = 0
        self.best_ae = 0  # None
        self.best_dis_ar = 0  # None
        self.best_val_loss = np.inf
        self.val_loss = 0  # None
        self.early_stop_count = 0
        self.re_loss = 0  # None

        self.time_per_epoch = 0  # None
        # loss list
        self.loss_normal_list = []
        # self.loss_abnormal_list = []
        # self.EMA_coefficient=0

    def train(self):  # 主程序
        logging.info('*' * 20 + 'Start training' + '*' * 20)
        for i in range(self.epoch):
            self.cur_epoch += 1
            if self.lr_scheduler is not None:
                # self.lr_scheduler.step(epoch)
                logging.info('current lr: {}'.format(self.lr_scheduler.get_last_lr()))
            else:
                logging.info('current lr: {}'.format(self.args.lr))
            self.train_epoch()  # 训练一次
            # logging.info('train time:{}'.format(self.time_per_epoch))
            self.validate()  # 验证一次

            if self.val_loss < self.best_val_loss and self.best_val_loss - self.val_loss >= 1e-4:
                self.best_val_loss = self.val_loss
                # self.best_ae = deepcopy(self.ae)
                self.best_epoch = i + 1
                # if self.args.save_model:
                self.save_best_model(self.ae)
                self.early_stop_count = 0
            elif self.early_stop:  # 早停, return结束程序：验证loss比最好loss高或者几乎一样连续累计n次后结束程序。只看重构误差
                self.early_stop_count += 1
                if self.early_stop_count > self.early_stop_tol:
                    logging.info('*' * 20 + 'Early stop' + '*' * 20)
                    return
            else:
                pass
            logging.info('[Epoch %d/%d] current training loss is %.5f, val loss is %.5f'
                         ' time per epoch is %.5f' % (i + 1, self.epoch, self.re_loss, self.val_loss,
                                                      self.time_per_epoch))
        self.plot()

    def train_epoch(self):  # 主程序训练
        start_time = time()
        number_normal = 0.0
        normal_sum = 0.0

        for batch_idx, (x, label) in enumerate(self.data_loader['train']):
            self.cur_step += 1
            x = torch.unsqueeze(x.to(self.device),dim=3)
            label = label.to(self.device)
            number_normal += x.size()[0]#样本数量
            loss_tmp= self.ae_train(x, label)  # 主程序训练生成器
            normal_sum += loss_tmp.item()
        # one epoch add loss to list
        self.loss_normal_list.append(normal_sum / number_normal)
        end_time = time()
        self.time_per_epoch = end_time - start_time
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

    def ae_train(self, x, label):  # 主程序训练生成器：对抗loss加上根据重构误差自适应的加权MSE的loss
        self.ae_optimizer.zero_grad()
        # sigPred, coeff, lossFilter
        re_x, z, lossSparsity = self.ae(x)  # 重构变量和中间变量
        # batch_numer, channel_number, seq_length = x.size()
        # base = channel_number * seq_length
        sparsityloss=torch.mean(lossSparsity)
        self.re_loss = self.mse(re_x, x) + self.sparsitylosscoeff*sparsityloss

        # backpropagation
        # self.EMA_coefficient = 0.9*self.EMA_coefficient+0.1*torch.mean(concate_coeff(z),dim=0).detach()#移动平均获取系数特征
        self.re_loss.backward(retain_graph=True)
        # self.re_loss.backward()
        self.ae_optimizer.step()

        return self.re_loss

    def validate(self):
        self.ae.eval()  # 验证模式
        self.val_loss = 0
        num_batch = 0
        for batch_idx, (x, _) in enumerate(self.data_loader['val']):
            x = torch.unsqueeze(x.to(self.device),dim=3)
            re_values = self.value_reconstruction_val(x)
            self.val_loss += self.mse(x, re_values).item() * x.size(0)
            num_batch += x.size(0)
        self.val_loss = self.val_loss / num_batch
        self.ae.train()  # 训练模式

    def test(self, load_from_file=False, last_epoch=False,titleadd=''):  # 测试集测试
        if load_from_file:
            logging.info('load from file, the best model in the epoch: {}'.format(self.best_epoch))
            model = self.load_best_model()
        else:
            logging.info('last epoch\'s model performance')
            model = self.ae
        model.eval()
        values_list = []
        labels_list = []
        re_values_list = []

        for batch_idx, (test_x, test_y) in enumerate(self.data_loader['test']):
            test_x = torch.unsqueeze(test_x.to(self.device),dim=3)
            re_values, _,_ = model(test_x)
            labels_list += test_y.numpy().tolist()
            values_list += test_x.cpu().numpy().tolist()
            re_values_list += re_values.detach().cpu().numpy().tolist()

        values_a = np.array(values_list)
        values_a = values_a.squeeze()
        re_values_a = np.array(re_values_list)
        re_values_a = re_values_a.squeeze()
        labels_a = np.array(labels_list)
        metrics_calculate(values_a, re_values_a, labels_a, self.plot_confusion_matrix,
                          title= self.data_prefix + titleadd, savedir=self.savedir)

        if self.args.save_reconstructed_data:
            self.save_result(values_a, re_values_a, labels_a)


    def value_reconstruction_val(self, raw_values, val=True):
        '''
        if train(val): reconstruct with current model
        if validate:   reconstruct with best model
        '''
        if val:
            reconstructed_value_, z,_ = self.ae(raw_values)
        else:
            reconstructed_value_, z,_ = self.best_ae(raw_values)
        return reconstructed_value_

    def test_hidden(self, load_from_file=False, metric_districution=False, feature_tsne=True, last_epoch=False):
        '''
        test hidden features
        plot tsne for hidden features
        '''
        if load_from_file:
            logging.info('load from file, the best model in the epoch: {}'.format(self.best_epoch))
            model = self.load_best_model()
        elif last_epoch:
            logging.info('last epoch\'s model performance')
            model = self.ae
        model.eval()
        values_list = []
        labels_list = []
        re_values_list = []

        for batch_idx, (test_x, test_y) in enumerate(self.data_loader['test']):
            test_x = torch.unsqueeze(test_x.to(self.device),dim=3)
            _, re_values,_ = model(test_x)  # 隐层特征
            zero = torch.zeros(re_values.size())
            labels_list += test_y.numpy().tolist()
            values_list += zero.numpy().tolist()
            re_values_list += re_values.detach().cpu().numpy().tolist()

        values_a = np.array(values_list)  # 0
        values_a = values_a.squeeze()
        re_values_a = np.array(re_values_list)  # 隐层特征
        re_values_a = re_values_a.squeeze()
        labels_a = np.array(labels_list)
        if metric_districution:
            metrics_calculate(values_a, re_values_a, labels_a, self.plot_confusion_matrix,
                              title='hidden'  + '_' + self.data_prefix)
        if feature_tsne:
            T_SNE(re_values_a, labels_a,dim=2)

    def save_best_model(self, model):
        if torch.cuda.device_count() > 1:
            torch.save(model.module.state_dict(), os.path.join(self.savedir,
                                                               self.args.model_prefix + '.pth'))
        else:
            torch.save(model.state_dict(), os.path.join(self.savedir,
                                                        self.args.model_prefix + '.pth'))

    def load_best_model(self):
        if self.args.DWTInit==1:
            kernelInit = np.array([-0.010597401785069032, 0.0328830116668852, 0.030841381835560764, -0.18703481171909309,
                               -0.027983769416859854, 0.6308807679298589, 0.7148465705529157, 0.2303778133088965])
        else:
            kernelInit = self.args.DWTInit
        ae = DeSpaWN(kernelInit=kernelInit, kernTrainable=self.args.kernTrainable, level=self.args.DWTlevel,
                     lossCoeff='l1', kernelsConstraint='PerLayer', initHT=1.0, trainHT=True)


        ae.load_state_dict(
            torch.load(os.path.join(self.savedir, self.args.model_prefix + '.pth')))
        ae = ae.to(self.device)
        if self.device_count > 1:
            ae = torch.nn.DataParallel(ae)
        return ae

    def save_result(self, values, re_values, labels):
        filename = os.path.join(self.savedir, 'reconstruction_data.mat')
        scipy.io.savemat(filename, {'values': values, 're_values': re_values, 'labels': labels})


    def plot(self):
        x = range(0, self.epoch)
        plt.title('loss in traning stage_'  + '_' + self.data_prefix)
        plt.plot(x, self.loss_normal_list, color='black')#, label='normal_loss'
        # plt.plot(x, self.loss_abnormal_list, color='red', label='abnormal_loss')
        plt.ylabel('Loss')
        plt.xlabel('epochs')
        # plt.legend()
        plt.show()
        scipy.io.savemat(os.path.join(self.savedir, 'loss.mat'),
                         {'normal': self.loss_normal_list})

    def plot_reconstruction_trainset(self, figure_number=1, channel=0, test=False):  # 可视化 训练集 重构效果图 只有正常
        self.ae.eval()
        first_batch = self.data_loader['train'].__iter__().__next__()
        sample = first_batch[0]
        reconstruct_sample, _,_ = self.ae(torch.unsqueeze(sample.to(self.device),dim=3))
        reconstruct_sample = reconstruct_sample.detach().cpu().numpy()

        for i in range(figure_number):
            plt.title('train set Normal:'+ '_' + self.data_prefix)
            plt.plot(sample[i, channel, :], color='black', label='original')
            plt.plot(reconstruct_sample[i, channel, :,0], color='red', label='reconstructed')
            plt.legend()
            plt.show()

    def plot_reconstruction_testset(self, figure_number=1, channel=0):  # 可视化 测试集 重构效果图 头部正常 尾部异常
        batch_number = len(self.data_loader['test'])
        if batch_number > 1:
            for batch_idex, (x, y) in enumerate(self.data_loader['test']):
                if batch_idex == 0:
                    first_batch_label = y
                    sample = x.detach().numpy()
                    reconstruct_sample, _,_ = self.ae(torch.unsqueeze(x.to(self.device),dim=3))
                    reconstruct_sample = reconstruct_sample.detach().cpu().numpy()
                elif batch_idex == batch_number - 1:
                    sample_last = x.detach().numpy()
                    last_batch_label = y
                    reconstruct_sample_last, _,_ = self.ae(torch.unsqueeze(x.to(self.device),dim=3))
                    reconstruct_sample_last = reconstruct_sample_last.detach().cpu().numpy()

            for i in range(figure_number):  # 头部的重构效果
                if first_batch_label[i] == 0:
                    plt.title('test set Normal:'  + '_' + self.data_prefix)
                else:
                    plt.title('test set Abnormal:'  + '_' + self.data_prefix)
                plt.plot(sample[i, channel, :], color='black', label='original signal')
                plt.plot(reconstruct_sample[i, channel, :,0], color='red', label='reconstructed')
                plt.legend()
                plt.show()

            if figure_number > last_batch_label.numel():
                figure_number = last_batch_label.numel()
            else:
                figure_number = figure_number

            for i in range(figure_number):  # 尾部的重构效果
                if last_batch_label[i] == 0:
                    plt.title('test set Normal:'  + '_' + self.data_prefix)
                else:
                    plt.title('test set Abnormal:'  + '_' + self.data_prefix)
                plt.plot(sample_last[i, channel, :], color='black', label='original signal')
                plt.plot(reconstruct_sample_last[i, channel, :,0], color='red', label='reconstructed')
                plt.legend()
                plt.show()
        else:
            for batch_idex, (x, y) in enumerate(self.data_loader['test']):
                if batch_idex == 0:
                    first_batch_label = y
                    sample = x.detach().numpy()
                    reconstruct_sample, _, _ = self.ae(torch.unsqueeze(x.to(self.device), dim=3))
                    reconstruct_sample = reconstruct_sample.detach().cpu().numpy()

            for i in range(figure_number):  # 头部的重构效果
                if first_batch_label[i] == 0:
                    plt.title('test set Normal:' + '_' + self.data_prefix)
                else:
                    plt.title('test set Abnormal:' + '_' + self.data_prefix)
                plt.plot(sample[i, channel, :], color='black', label='original signal')
                plt.plot(reconstruct_sample[i, channel, :, 0], color='red', label='reconstructed')
                plt.legend()
                plt.show()

            for i in range(figure_number):  # 尾部的重构效果
                if first_batch_label[-i] == 0:
                    plt.title('test set Normal:' + '_' + self.data_prefix)
                else:
                    plt.title('test set Abnormal:' + '_' + self.data_prefix)
                plt.plot(sample[-i, channel, :], color='black', label='original signal')
                plt.plot(reconstruct_sample[-i, channel, :, 0], color='red', label='reconstructed')
                plt.legend()
                plt.show()