import scipy.io
import torch
import torch.optim
import numpy as np
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
# import visdom
# import mkdir
import time
# torch.manual_seed(1337)
# np.random.seed(1337)
# torch.cuda.manual_seed(1337)






class DanQ(nn.Module):
    def __init__(self, config,):
        super(DanQ, self).__init__()
        '''NOT currently in use.'''
        self.config = config

        self.Conv1 = nn.Conv1d(in_channels=4, out_channels=320, kernel_size=26)
        #self.Conv1.weight.data = torch.Tensor(np.load('conv1_weights.npy'))
        #self.Conv1.bias.data = torch.Tensor(np.load('conv1_bias.npy'))
        self.Maxpool = nn.MaxPool1d(kernel_size=13, stride=13)
        self.Drop1 = nn.Dropout(p=0.2)
        self.BiLSTM = nn.LSTM(input_size=320, hidden_size=320, num_layers=2,
                                 batch_first=True,
                                 dropout=0.5,
                                 bidirectional=True)
        self.Linear1 = nn.Linear(10880, 925)
        self.Linear2 = nn.Linear(925, 2)
        self.optim = torch.optim.RMSprop(self.parameters(), lr=self.config['lr'])
        self.sched = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optim, patience=5, verbose=1)

    def forward(self, h):
        h = self.Conv1(h)
        h = F.relu(h)
        h = self.Maxpool(h)
        h = self.Drop1(h)
        h = torch.transpose(h, 1, 2)
        h, (h_n,h_c) = self.BiLSTM(h)
        #h, h_n = self.BiGRU(h)
        # h = h.contiguous().view(-1, 75*640)
        # print('h shape 0 {}'.format(h.shape, ))
        h = h.reshape(h.shape[0], -1)
        # print('h shape 5 {}'.format(h.shape, ))

        h = self.Linear1(h)
        h = F.relu(h)
        h = self.Linear2(h)
        return h