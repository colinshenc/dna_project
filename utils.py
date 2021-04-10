# -*- coding: utf-8 -*-




from __future__ import print_function
import sys
import os
import numpy as np
import time
import datetime
import json
import pickle
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import matplotlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import pandas as pd

def prepare_parser():
    usage = 'Parser for all scripts.'
    parser = ArgumentParser(description=usage)

    parser.add_argument(
        '--model', type=str, default='',
        help='Which model to use (default: %(default)s)'
    )
    ###Experiment###
    parser.add_argument(
        '--num_epochs', type=int, default=10,
        help='Number of epochs to train for (default: %(default)s)')

    parser.add_argument(
        '--kernel_size', type=int, default=19,
        help='Number of kernels (default: %(default)s)')
    parser.add_argument(
        '--mpool_kernel_size', type=int, default=3,
        help='Number of maxpool kernels (default: %(default)s)')
    parser.add_argument(
        '--mpool_stride', type=int, default=3,
        help='Number of maxpool kernels (default: %(default)s)')
    parser.add_argument(
        '--batch_size', type=int, default=128,
        help='Default overall batch size (default: %(default)s)')
    parser.add_argument(
        '--lr', type=float, default=1e-5,
        help='Learning rate to use for Generator (default: %(default)s)')
    parser.add_argument(
        '--data_root', type=str, default='/ubc/cs/research/shield/projects/cshen001/dna_project/data/',
        help='Default location where data is stored (default: %(default)s)')
    parser.add_argument(
        '--data_file', type=str, default='dataset_chr_left_out_1_05_valid.h5',
        help='Name of the file in data path (default: %(default)s)')
    parser.add_argument(
        '--ckpts_path', type=str, default='/ubc/cs/research/shield/projects/cshen001/dna_project/human_mouse_dna_classification/ckpts/',
        help='Name of the checkpoints file: %(default)s)')
    parser.add_argument(
        '--graphs_path', type=str, default='/ubc/cs/research/shield/projects/cshen001/dna_project/human_mouse_dna_classification/graphs/',
        help='Name of the checkpoints file: %(default)s)')

    parser.add_argument(
        '--in_channels', type=int, default=4,
        help='Number of input data channels(default: %(default)s)')
    parser.add_argument(
        '--out_channels', type=int, default=2,
        help='Number of outpuy classes(default: %(default)s)')

    parser.add_argument(
        '--feature_multiplier', type=int, default=96,
        help='Number of ground feautures (default: %(default)s)')

    parser.add_argument(
        '--resume', action='store_true', default=False,
        help='Resume training? (default: %(default)s)')

    parser.add_argument(
        '--exp_name', type=str, default='',
        help='Optionally override the automatic experiment naming with this arg. '
             '(default: %(default)s)')
    parser.add_argument(
        '--init_func', type=str, default='ortho',
        help='Init style to use for the model (default: %(default)s)')

    return parser

def toggle_gradients(model, on_or_off):
  for param in model.parameters():
    param.requires_grad = on_or_off


# Save a model's weights, optimizer, and the state_dict
#Modified from BigGAN code
def save_weights(config, model, state_dict,
                 ):
    # model.eval()
    #root = '/'.join([weights_root, experiment_name])
    path = config['ckpts_path']
    experiment_name = config['exp_name']
    # best_epoch = state_dict['best_epoch']
    if not os.path.exists(path):
        print('Checkpoints path does not exist, making new path...')
        os.mkdir(path)

    print('Saving weights to {}...'.format(path[70:]))
    torch.save(model.state_dict(),
              '{}/{}_model.pth'.format(path, experiment_name, ))
    torch.save(model.optim.state_dict(),
              '{}/{}_optim.pth'.format(path, experiment_name,))
    torch.save(state_dict,
              '{}/{}_state_dict.pth'.format(path, experiment_name,))
    # model.train()


# Load a model's weights, optimizer, and the state_dict
# Modified from BigGAN's code
def load_weights(config, model, state_dict,
                 strict=True):
    path = config['ckpts_path']
    experiment_name = config['exp_name']

    print('Loading weights from {}...'.format(path))

    model.load_state_dict(
      torch.load('{}/{}_model.pth'.format(path, experiment_name)),
      strict=strict)
    #if load_optim:
    model.optim.load_state_dict(
        torch.load('{}/{}_optim.pth'.format(path, experiment_name)))

    # Load state dict
    for item in state_dict:
        state_dict[item] = torch.load('{}/{}_state_dict.pth'.format(path, experiment_name))[item]

def plot():
    with open('/ubc/cs/research/shield/projects/cshen001/dna_project/human_mouse_dna_classification/ckpts/08-23-2020-11:41:42_model_ours_bs_100_lr_0.003_0_with_Maxpool_data_for_plot.txt') as file0_mp, \
    open('/ubc/cs/research/shield/projects/cshen001/dna_project/human_mouse_dna_classification/ckpts/08-23-2020-11:41:42_model_ours_bs_100_lr_0.003_1_with_Maxpool_data_for_plot.txt') as file1_mp, \
    open('/ubc/cs/research/shield/projects/cshen001/dna_project/human_mouse_dna_classification/ckpts/08-23-2020-11:41:42_model_ours_bs_100_lr_0.003_2_with_Maxpool_data_for_plot.txt') as file2_mp:
    # open('/ubc/cs/research/shield/projects/cshen001/dna_project/human_mouse_dna_classification/ckpts/08-04-2020-23:46:10_feat_mult_16_bs_2048_lr_2e-05_0_Maxpool_data_for_plot.txt') as file0_mp, \
    # open('/ubc/cs/research/shield/projects/cshen001/dna_project/human_mouse_dna_classification/ckpts/08-04-2020-23:46:10_feat_mult_16_bs_2048_lr_2e-05_1_Maxpool_data_for_plot.txt') as file1_mp, \
    # open('/ubc/cs/research/shield/projects/cshen001/dna_project/human_mouse_dna_classification/ckpts/08-04-2020-23:46:10_feat_mult_16_bs_2048_lr_2e-05_2_Maxpool_data_for_plot.txt') as file2_mp:
    #
    #     plot_dict_0_fm = json.load(file0_feat_mult)
    #     plot_dict_1_fm = json.load(file1_feat_mult)
    #     plot_dict_2_fm = json.load(file2_feat_mult)

        plot_dict_0_mp = json.load(file0_mp)
        plot_dict_1_mp = json.load(file1_mp)
        plot_dict_2_mp = json.load(file2_mp)

    feat_mult = [300,320,384,512,768]#[8,16,48,96,128,192,256,300,320]#[8,96,192,256,300,320,360,384,512]#[8, 16, 48, 96, 128, 192, 256, 300, 320, 360, 420]
    fig = plt.figure()
    # ax = fig.add_axes([0, 0, 1, 1])
    #fm0 = pd.DataFrame({'feat_mult':feat_mult* 6, 'score':plot_dict_0_fm['roc_auc'] + plot_dict_0_fm['auprc'] + plot_dict_1_fm['roc_auc']+plot_dict_1_fm['auprc']+plot_dict_2_fm['roc_auc']+plot_dict_2_fm['auprc'], 'Metrics':['roc_auc0']*6 +['auprc0']*6+['roc_auc1']*6 +['auprc1']*6+['roc_auc2']*6 +['auprc2']*6})#.rename(index={0:'roc_auc',1:'auprc'},inplace=True)
    # sns.set(style="white")
    # plot_dict_0_womp = {'roc_auc':[0.80,0.83,0.84,0.85,0.85],}
    # plot_dict_1_womp =
    # plot_dict_2_womp =

    # Plot miles per gallon against horsepower with other semantics
    # plot = sns.lineplot(x='feat_mult', y='score',
    #              alpha=1, palette="muted", hue='Metrics', marker=['*']*6+['.']*6+['v']*6+['^']*6+['p']*6+['h']*6, color=['red']*12+['blue']*12+['green']*12,
    #              data=fm0)
    plt.plot(feat_mult, plot_dict_0_mp['roc_auc'], color='g', marker='*', label='roc_auc0', linestyle='-.',alpha=0.65)
    plt.plot(feat_mult, plot_dict_0_mp['auprc'], color='g', marker='.', label='auprc0', linestyle='-.',alpha=0.65)
    plt.plot(feat_mult, plot_dict_1_mp['roc_auc'], color='c', marker='v', label='roc_auc1', linestyle='-', alpha=0.65)
    plt.plot(feat_mult, plot_dict_1_mp['auprc'], color='c', marker='^', label='auprc1', linestyle='-', alpha=0.65)
    plt.plot(feat_mult, plot_dict_2_mp['roc_auc'], color='y', marker='p', label='roc_auc2', linestyle=':', alpha=0.65)
    plt.plot(feat_mult, plot_dict_2_mp['auprc'], color='y', marker='h', label='auprc2', linestyle=':', alpha=0.65)
    # matplotlib.rcParams['font.sans-serif'] = 'Cambria'
    #matplotlib.rcParams['font.family'] = "sans-serif"
    # plt.plot(feat_mult, plot_dict_0_fm['auprc'], color='g', marker='*', label='auprc0_wo_maxpool', linestyle='-.', alpha=0.65)
    # plt.plot(feat_mult, plot_dict_1_fm['auprc'], color='y', marker='v', label='auprc1_wo_maxpool', linestyle='-', alpha=0.65)
    # plt.plot(feat_mult, plot_dict_2_fm['auprc'], color='m', marker='p', label='auprc2_wo_maxpool', linestyle=':', alpha=0.65)

    plt.xticks(feat_mult)
    # plt.xticklabels(feat_mult)
    plt.legend(loc="lower right")
    plt.xlabel('feature multiplier')
    plt.ylabel('score')
    plt.title('Number of output_channels (first layer) range results(three\n identical experiments original model)')

    plt.savefig('/ubc/cs/research/shield/projects/cshen001/dna_project/human_mouse_dna_classification/ckpts/after_debug/{}.jpg'.format('08-23-2020-11:41:42_ours_extra'), dpi=600)
    # # columns=plot_dict_0_fm.keys())

    # print(fm0)
# def no_scheduler(*args, **kwargs):
#     return