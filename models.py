import torch.nn.functional as F
import torch.utils.data
import torch.nn as nn
from torch import relu, sigmoid
from collections import OrderedDict
import torch.nn.modules.activation as activation
from torch.nn import init

#the deep learning model (Basset architecture)
class ConvNetDeepCrossSpecies(nn.Module):
    def __init__(self, config,) :
        super(ConvNetDeepCrossSpecies, self).__init__()
        # Block 1 :
        self.weight_path = config['ckpts_path']
        self.in_channels = config['in_channels']
        self.out_channels = config['out_channels']
        self.feat_mult = config['feature_multiplier']
        self.init = config['init_func']
        self.kernel_size = config['kernel_size']
        self.maxpool_kernel_size = config['mpool_kernel_size']
        self.maxpool_stride = config['mpool_stride']
        #self.emb = nn.Embedding(12, 4)

        # self.l0 = nn.Linear(1000, 256)
        # self.bn0 = nn.BatchNorm1d(256)
        # self.rl0 = nn.LeakyReLU()

        self.c1 = nn.Conv1d(in_channels=self.in_channels, out_channels=self.feat_mult, kernel_size=self.kernel_size)
        self.bn1 = nn.BatchNorm1d(self.feat_mult)
        self.rl1 = nn.ReLU()
        # self.mp1 = nn.MaxPool1d(self.maxpool_kernel_size, stride=self.maxpool_stride)

        #self.resblock1 = BasicBlock1D(in_channels=self.feat_mult, out_channels=self.feat_mult, kernel_size=23)

        # Block 2 :
        self.c2 = nn.Conv1d(in_channels=self.feat_mult, out_channels=200, kernel_size=11)
        self.bn2 = nn.BatchNorm1d(200)
        self.rl2 = nn.ReLU()
        # self.mp2 = nn.MaxPool1d(self.maxpool_kernel_size, stride=self.maxpool_stride)
        #self.resblock2 = BasicBlock1D(in_channels=2*self.feat_mult, out_channels=2*self.feat_mult, kernel_size=23)

        # Block 3 :
        # self.c3 = nn.Conv1d(in_channels=2*self.feat_mult, out_channels=4*self.feat_mult, kernel_size=23)
        # self.bn3 = nn.BatchNorm1d(4*self.feat_mult)
        # self.rl3 = nn.LeakyReLU()
        # # #self.mp3 = nn.MaxPool1d(3,3)
        # self.resblock3 = BasicBlock1D(in_channels=4*self.feat_mult, out_channels=4*self.feat_mult, kernel_size=23)

        self.c3 = nn.Conv1d(in_channels=200, out_channels=200, kernel_size=7)
        self.bn3 = nn.BatchNorm1d(200)
        #self.rl4 = nn.LeakyReLU()
        self.rl3 = nn.ReLU()
        # self.mp3 = nn.MaxPool1d(self.maxpool_kernel_size, stride=self.maxpool_stride)


        #self.resblock4 = BasicBlock1D(in_channels=4*self.feat_mult, out_channels=16, kernel_size=9)

        # Block 4 : Fully Connected 1 :
        self.d4 = nn.Linear(43200, 1000) #no mp:43200, mp22:4600, mp33:1000, mp44:200
        self.bn4 = nn.BatchNorm1d(1000, True)
        self.rl4 = nn.ReLU()
        #self.rl_ = nn.LeakyReLU()
        self.dr4 = nn.Dropout(0.3)

        self.d5 = nn.Linear(1000, 1000)  # 1000 for 200 input size
        self.bn5 = nn.BatchNorm1d(1000, True)
        self.rl5 = nn.ReLU()
        self.dr5 = nn.Dropout(0.3)

        # Block 5 : Fully Connected 2 :
        self.d6 = nn.Linear(1000, self.out_channels)
        # self.bn5 = nn.BatchNorm1d(self.out_channels,  True)
        # self.rl5 = nn.ReLU()
        #self.rl5 = nn.LeakyReLU()
        #self.dr5 = nn.Dropout(0.3)

        # Block 6 :4Fully connected 3
        #self.d6 = nn.Linear(256, self.out_channels)
        #self.sig = activation.Sigmoid()
        self.optim = torch.optim.Adam(params=self.parameters(), lr=config['lr'])
                                #betas=(self.B1, self.B2), weight_decay=0,
                                #eps=self.adam_eps)
        # if self.weight_path:
        #     self.load_weights(self.weight_path)
        if not config['resume'] and self.training:
            self.init_weights()

    def forward(self, x, embeddings=False):
        """
            :param: embeddings : if True forward return embeddings along with the output
        """
        # Block 1
        # x is of size - batch, 4, 200
        # print('0 shape {}'.format(x.shape))
        #print('-10 shape {}'.format(x.shape))
        #h = self.emb(x.long())
        # print('5 shape {}'.format(h.shape))
        #h = h.reshape(h.shape[0], h.shape[1], -1)
        # print('10 shape {}'.format(h.shape))
        #h = self.rl0(self.l0(h))
        h = self.rl1(self.bn1(self.c1(x))) # output - batch, 100, 182
        #print('-5 shape {}'.format(x.shape))
        #h = self.resblock1(h)
        #h = self.resblock3(h)
        #o = self.resblock4(h)
        # we save the activations of the first layer (interpretation)
        #activations = x # batch, 100, 182
        # h = self.mp1(h)

        # Block 2
        # input is of size batch, 100, 60
        #x = self.mp2(self.rl2(self.bn2(self.c2(x)))) #output - batch, 200, 18
        h = self.rl2(self.bn2(self.c2(h))) #output - batch, 200, 18
        #h = self.resblock2(h)
        # h = self.mp2(h)
        # Block 3
        # input is of size batch, 200, 18
        #em = self.mp3(self.rl3(self.bn3(self.c3(x)))) #output - batch, 200, 5
        #h = self.rl3(self.bn3(self.c3(h))) #output - batch, 200, 5
        #h = self.resblock3(h)

        h = self.c3(h)
        h = self.bn3(h)
        h = self.rl3(h)
        # h = self.mp3(h)
        #print('5 shape {}'.format(o.shape))
        # Flatten
        h = torch.flatten(h, start_dim=1) #output - batch, 1000
        #print('10 shape {}'.format(o.shape))

        # FC1
        #input is of size - batch, 1000
        h = self.dr4(self.rl4(self.bn4(self.d4(h)))) #output - batch, 1000

        # o = self.dr_(self.rl_(self.bn_(self.d_(o)))) #output - batch, 1000

        # FC2
        #input is of size - batch, 1000
        #o = self.dr5(self.rl5(self.bn5(self.d5(o)))) #output - batch, 1000
        #h = self.rl5(self.bn5(self.d5(h))) #output - batch, 1000
        h = self.dr5(self.rl5(self.bn5(self.d5(h))))        # FC3
        #input is of size - batch, 1000
        #o = self.sig(self.d6(o)) #output - batch, num_of_classes
        # o = self.d6(o) #doing BCEWithLogits #output - batch, num_of_classes

        #o = torch.tanh(o) #use hinge loss
        #print('output shape {}'.format(o.shape))
        #maximum for every filter (and corresponding index)
        #activations, act_index = torch.max(activations, dim=2)
        h = self.d6(h)
        if embeddings:
            return(h, activations, act_index, em)
        #return (o, activations, act_index)
        # print(o.shape)
        # print(o[1])
        return h

    # Initialize
    def init_weights(self):
        self.param_count = 0
        for module in self.modules():
            if (isinstance(module, nn.Conv1d)
                or isinstance(module, nn.Linear)
                or isinstance(module, nn.Embedding)):
                if self.init == 'ortho':
                    init.orthogonal_(module.weight)
                    print('init')
                elif self.init == 'N02':
                    init.normal_(module.weight, 0, 0.02)
                elif self.init in ['glorot', 'xavier']:
                    init.xavier_uniform_(module.weight)
                elif self.init == 'no_init':
                    print('no init')
                    continue
                else:
                    print('Init style not recognized...')
                self.param_count += sum([p.data.nelement() for p in module.parameters()])
        print('Param count for model''s initialized parameters: %d' % self.param_count)

    #
    # def load_weights(self, weight_path):
    #     print('Loading weights...')
    #     sd = torch.load(weight_path)
    #     print(sd)
    #     new_dict = OrderedDict()
    #     keys = list(self.state_dict().keys())
    #     values = list(sd.values())
    #     for i in range(len(values)):
    #         v = values[i]
    #         if v.dim() > 1 :
    #             if v.shape[-1] ==1 :
    #                 new_dict[keys[i]] = v.squeeze(-1)
    #                 continue
    #         new_dict[keys[i]] = v
    #     self.load_state_dict(new_dict)


#taken from pytorch resnet code
# class BasicBlock1D(nn.Module):
#     expansion = 1
#
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, downsample=None, groups=1,
#                  base_width=64, dilation=1, norm_layer=None):
#         super(BasicBlock1D, self).__init__()
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm1d
#         # if groups != 1 or base_width != 64:
#         #     raise ValueError('BasicBlock only supports groups=1 and base_width=64')
#         # if dilation > 1:
#         #     raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
#         # Both self.conv1 and self.downsample layers downsample the input when stride != 1
#         #self.in_channels = in_channels
#         #self.out_channels = out_channels
#         #self.kernel_size = kernel_size
#         self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
#         self.bn1 = norm_layer(out_channels)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size)
#         self.bn2 = norm_layer(out_channels)
#         self.downsample = nn.MaxPool1d(kernel_size=2*(kernel_size)-1, stride=1)
#         self.stride = stride
#
#     def forward(self, x):
#         identity = x
#         #print('shape 0 {}'.format(x.shape))
#         out = self.conv1(x)
#         #print('shape 5 {}'.format(out.shape))
#
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         #print('shape 10 {}'.format(out.shape))
#
#         out = self.bn2(out)
#
#         if self.downsample is not None:
#             identity = self.downsample(x)
#         #print('id shape {}'.format(identity.shape))
#         out = out + identity
#         out = self.relu(out)
#
#         return out
