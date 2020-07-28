import time
from tqdm import tqdm
import numpy as np
import pandas as pd
import collections
from numpy import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score, classification_report, confusion_matrix
from sklearn.metrics import matthews_corrcoef
from collections import OrderedDict
import os
import pickle
from models import *
import utils
import torch
import torch.nn.functional as F
import torch.utils.data
import torch.nn as nn
from torch import relu, sigmoid
import torch.nn.modules.activation as activation
import matplotlib

matplotlib.use('Agg')
from scipy.spatial import distance
from scipy.spatial.distance import cdist
from sklearn import metrics
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import matplotlib.ticker as ticker
import copy

import h5py
### import kipoi
#import seaborn as sns

############################################################
#function for loading the dataset
############################################################
def get_data_loader(config, ):
    data_path = '{}{}'.format(config['data_root'], config['data_file'])
    print('\n\n00000')
    data = h5py.File(data_path, 'r')

    dataset = {}
    dataloaders = {}
    #Train data
    dataset['train'] = torch.utils.data.TensorDataset(torch.Tensor(data['train_in']), 
                                                     torch.Tensor(data['train_out']))
    print('train length {}'.format(len(dataset['train'])))
    dataloaders['train'] = torch.utils.data.DataLoader(dataset['train'],
                                                      batch_size=config['batch_size'], shuffle=True,
                                                      num_workers=12, pin_memory=True, drop_last=False,)
    
    #Validation data
    dataset['valid'] = torch.utils.data.TensorDataset(torch.Tensor(data['valid_in']), 
                                                     torch.Tensor(data['valid_out']))
    print('valid length {}'.format(len(dataset['valid'])))

    dataloaders['valid'] = torch.utils.data.DataLoader(dataset['valid'],
                                                      batch_size=config['batch_size'], shuffle=True, pin_memory=True, drop_last=False,
                                                      num_workers=12)
    
    #Test data
    dataset['test'] = torch.utils.data.TensorDataset(torch.Tensor(data['test_in']), 
                                                     torch.Tensor(data['test_out']))
    print('test length {}'.format(len(dataset['test'])))

    dataloaders['test'] = torch.utils.data.DataLoader(dataset['test'],
                                                      batch_size=config['batch_size'], shuffle=True, pin_memory=True, drop_last=False,
                                                     num_workers=12)
    print('Dataset Loaded')
    target_labels = list(data['target_labels'])
    train_out = data['train_out']
    return dataloaders, target_labels, train_out

############################################################
#function to convert sequences to one hot encoding
#taken from Basset github repo
############################################################
def dna_one_hot(seq, seq_len=None, flatten=True):
    if seq_len == None:
        seq_len = len(seq)
        seq_start = 0
    else:
        if seq_len <= len(seq):
            # trim the sequence
            seq_trim = (len(seq)-seq_len) // 2
            seq = seq[seq_trim:seq_trim+seq_len]
            seq_start = 0
        else:
            seq_start = (seq_len-len(seq)) // 2

    seq = seq.upper()

    seq = seq.replace('A','0')
    seq = seq.replace('C','1')
    seq = seq.replace('G','2')
    seq = seq.replace('T','3')

    # map nt's to a matrix 4 x len(seq) of 0's and 1's.
    #  dtype='int8' fails for N's
    seq_code = np.zeros((4,seq_len), dtype='float16')
    for i in range(seq_len):
        if i < seq_start:
            seq_code[:,i] = 0.25
        else:
            try:
                seq_code[int(seq[i-seq_start]),i] = 1
            except:
                seq_code[:,i] = 0.25

    # flatten and make a column vector 1 x len(seq)
    if flatten:
        seq_code = seq_code.flatten()[None,:]

    return seq_code

############################################################
#function to train a model
############################################################
def train_model(config, train_loader, test_loader, model, device, criterion, state_dict,
             verbose):
    
    #total_step = len(train_loader)
    
    train_error = []
    test_error = []
    
    #train_fscore = []
    #test_fscore = []

    # best_model_wts = copy.deepcopy(model.state_dict())
    # best_loss_valid = float('inf')
    # best_epoch = 1
    # ckpt_path = '{}{}'.format(config[''])
    for epoch in range(config['num_epochs']):
        
        model.train() #tell model explicitly that we train
        utils.toggle_gradients(model, True)
        #logs = {}
        
        running_loss = 0.0
        #running_fbeta = 0.0
        
        for seqs, labels in train_loader:
            x = seqs.to(device)
            labels = labels.to(device)
            
            #zero the existing gradients so they don't add up
            model.optim.zero_grad()

            # Forward pass
            outputs = model(x)
            loss = criterion(outputs, labels) 
            
            # Backward and optimize
            loss.backward()
            model.optim.step()
            
            running_loss = running_loss + loss.item()
            
        #scheduler.step() #learning rate schedule
        
        #save training loss to file
        epoch_loss = running_loss / len(train_loader)
        #logs['train_log_loss'] = epoch_loss
        train_error.append(epoch_loss)
        

        #calculate test (validation) loss for epoch
        test_loss = 0.0
        #test_fbeta = 0.0
        
        with torch.no_grad(): #we don't train and don't save gradients here
            model.eval() #we set forward module to change dropout and batch normalization techniques
            for seqs, labels in test_loader:
                x = seqs.to(device)
                y = labels.to(device)
                model.eval() #we set forward module to change dropout and batch normalization techniques
                outputs = model(x)
                loss = criterion(outputs, y)
                test_loss = test_loss + loss.item()

        test_loss = test_loss / len(test_loader) 
        #logs['test_log_loss'] = test_loss
        test_error.append(test_loss)
        
        if verbose:
            print_msg = 'Epoch [{}], Current Train Loss: {:.5f}, Current Val Loss: {:.5f}'.format(epoch, epoch_loss, test_loss)
            with open("{}{}_results_log.txt".format(config['ckpts_path'], config['exp_name']), "a+") as file:
                file.write(print_msg)

            print(print_msg)
        if test_loss < state_dict['best_test_loss']:
            print('Best epoch {}'.format(epoch))
            state_dict['best_test_loss'] = test_loss
            state_dict['best_epoch'] = epoch
            utils.save_weights(config, model, state_dict)
            # best_model_wts = copy.deepcopy(model.state_dict())
                 #name_ind+".pth") #weights_folder, name_ind

    # model.load_state_dict(best_model_wts)
    # torch.save(best_model_wts, config['ckpts_path'] + "/"+"model_epoch_"+str(best_epoch+1)+"_"+
    #                    name_ind+".pth") #weights_folder, name_ind
    
    #return model, best_loss_valid
    return model, train_error, test_error

############################################################    
#function to test the performance of the model
############################################################
def run_test( model, dataloader_test, device):
    print('Start testing...')
    #Switch it to eval mode
    model.eval()
    utils.toggle_gradients(model, False)

    running_outputs = torch.tensor([],dtype=torch.float32)
    running_labels = torch.tensor([],dtype=torch.float32)
    running_outputs_argmax = torch.tensor([], dtype=torch.int8)
    running_labels_argmax = torch.tensor([], dtype=torch.int8)
    #sigmoid = nn.Sigmoid()
    with torch.no_grad():
        for seq, lbl in dataloader_test:
            seq = seq.to(device)
            out = model(seq)
            #print('out 1 {}'.format(out.shape))
            out = nn.Softmax()(out.detach().cpu())
            #print('out 2 {}'.format(out[1]))
            #print('label {}'.format(lbl.shape))
            #out = out.detach().cpu() #for BCEWithLogits
            out_argmax = torch.argmax(out, dim=1, keepdim=False).to(torch.int8)
            lbl_argmax = torch.argmax(lbl, dim=1, keepdim=False).to(torch.int8)

            # print('out {}'.format(out.shape))
            # print('label {}'.format(lbl.shape))
            # print('out 10 {}'.format(out[:10]))
            # print('label 10 {}'.format(lbl[:10]))
            #
            # print('out am {}'.format(out_argmax.shape))
            # print('label am {}'.format(lbl_argmax.shape))
            # print('out 10 am {}'.format(out_argmax[:10]))
            # print('label 10 am {}'.format(lbl_argmax[:10]))

            running_outputs_argmax = torch.cat([running_outputs_argmax, out_argmax.to(torch.int8)]) #for BCEWithLogits
            running_labels_argmax = torch.cat([running_labels_argmax, lbl_argmax.to(torch.int8)])
            running_outputs = torch.cat([running_outputs, out.to(torch.float32)])  # for BCEWithLogits
            running_labels = torch.cat([running_labels, lbl.to(torch.float32)])
    return running_labels.numpy(), running_outputs.numpy(), running_labels_argmax.numpy(), running_outputs_argmax.numpy()

############################################################
#functions to compute the metrics
############################################################
def compute_metrics(config, labels, outputs, labels_argmax, outputs_argmax):
    # TP = np.sum(((labels == 1) * (np.round(outputs) == 1)))
    # FP = np.sum(((labels == 0) * (np.round(outputs) == 1)))
    # TN = np.sum(((labels == 0) * (np.round(outputs) == 0)))
    # FN = np.sum(((labels == 1) * (np.round(outputs) == 0)))
    # print('TP : {} FP : {} TN : {} FN : {}'.format(TP, FP, TN, FN))
    # plt.bar(['TP', 'FP', 'TN', 'FN'], [TP, FP, TN, FN])


    #plt.savefig('{}/{}.jpg'.format(config['graphs_path'], config['exp_name']))

    
    try:
        classification_report_ = classification_report(labels_argmax, outputs_argmax, target_names=['Mouse', 'Human'])
        roc_auc_score_ = 'Roc AUC Score : {:.2f}'.format(roc_auc_score(labels, outputs))
        auprc_ = 'AUPRC {:.2f}'.format(average_precision_score(labels, outputs))
        cm = confusion_matrix(labels_argmax, outputs_argmax)
        TN, FP, FN, TP = cm.ravel()
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        accuracy = (TP + TN) / (TP + FP + FN + TN)
        p_r_a = 'TP: {} TN: {} FP: {} FN: {}\n\nPrecision : {:.2f} Recall : {:.2f} Accuracy : {:.2f}'.format(TP, TN, FP, FN, precision, recall, accuracy)
        with open("{}{}_results_log.txt".format(config['ckpts_path'], config['exp_name']), "a+") as file:
            file.write('\n\n')
            file.write(classification_report_)
            file.write('\n')
            file.write(p_r_a)
            file.write('\n')
            file.write(str(cm))
            file.write('\n')
            file.write(roc_auc_score_)
            file.write('\n')
            file.write(auprc_)
            file.write('\n\n')

        print(p_r_a)
        print(classification_report_)
        print(roc_auc_score_)
        print(auprc_)
    except ValueError:
        print('value error!')
        pass
    # precision = TP / (TP + FP)
    # recall = TP / (TP + FN)
    # accuracy = (TP + TN) / (TP + FP + FN + TN)
    # print('Precision : {:.2f} Recall : {:.2f} Accuracy : {:.2f}'.format(precision, recall, accuracy))


