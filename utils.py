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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader



def prepare_parser():
    usage = 'Parser for all scripts.'
    parser = ArgumentParser(description=usage)

    ###Experiment###
    parser.add_argument(
        '--num_epochs', type=int, default=10,
        help='Number of epochs to train for (default: %(default)s)')
    parser.add_argument(
        '--batch_size', type=int, default=128,
        help='Default overall batch size (default: %(default)s)')
    parser.add_argument(
        '--lr', type=float, default=1e-3,
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
    model.eval()
    #root = '/'.join([weights_root, experiment_name])
    path = config['ckpts_path']
    experiment_name = config['exp_name']
    best_epoch = state_dict['best_epoch']
    if not os.path.exists(path):
        print('Checkpoints path does not exist, making new path...')
        os.mkdir(path)

    print('Saving weights to {}...'.format(path))
    torch.save(model.state_dict(),
              '{}/{}_model.pth'.format(path, experiment_name, ))
    torch.save(model.optim.state_dict(),
              '{}/{}_optim.pth'.format(path, experiment_name,))
    torch.save(state_dict,
              '{}/{}_state_dict.pth'.format(path, experiment_name,))
    model.train()


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
