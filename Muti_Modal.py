from torch import nn
import torch
import torchvision
import numpy as np
import math
import torch.nn.functional as F

import torch.nn.functional as F
from torch import nn
import torch

class GlobalLayerNorm(nn.Module):
    '''
       Calculate Global Layer Normalization
       dim: (int or list or torch.Size) –
          input shape from an expected input of size
       eps: a value added to the denominator for numerical stability.
       elementwise_affine: a boolean value that when set to True, 
          this module has learnable per-element affine parameters 
          initialized to ones (for weights) and zeros (for biases).
    '''

    def __init__(self, dim, shape, eps=1e-8, elementwise_affine=True):
        super(GlobalLayerNorm, self).__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            if shape == 3:
                self.weight = nn.Parameter(torch.ones(self.dim, 1))
                self.bias = nn.Parameter(torch.zeros(self.dim, 1))
            if shape == 4:
                self.weight = nn.Parameter(torch.ones(self.dim, 1, 1))
                self.bias = nn.Parameter(torch.zeros(self.dim, 1, 1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        # x = N x C x K x S or N x C x L
        # N x 1 x 1
        # cln: mean,var N x 1 x K x S
        # gln: mean,var N x 1 x 1
        if x.dim() == 4:
            mean = torch.mean(x, (1, 2, 3), keepdim=True)
            var = torch.mean((x-mean)**2, (1, 2, 3), keepdim=True)
            if self.elementwise_affine:
                x = self.weight*(x-mean)/torch.sqrt(var+self.eps)+self.bias
            else:
                x = (x-mean)/torch.sqrt(var+self.eps)
        if x.dim() == 3:
            mean = torch.mean(x, (1, 2), keepdim=True)
            var = torch.mean((x-mean)**2, (1, 2), keepdim=True)
            if self.elementwise_affine:
                x = self.weight*(x-mean)/torch.sqrt(var+self.eps)+self.bias
            else:
                x = (x-mean)/torch.sqrt(var+self.eps)
        return x


class CumulativeLayerNorm(nn.LayerNorm):
    '''
       Calculate Cumulative Layer Normalization
       dim: you want to norm dim
       elementwise_affine: learnable per-element affine parameters 
    '''

    def __init__(self, dim, elementwise_affine=True):
        super(CumulativeLayerNorm, self).__init__(
            dim, elementwise_affine=elementwise_affine, eps=1e-8)

    def forward(self, x):
        # x: N x C x K x S or N x C x L
        # N x K x S x C
        if x.dim() == 4:
           x = x.permute(0, 2, 3, 1).contiguous()
           # N x K x S x C == only channel norm
           x = super().forward(x)
           # N x C x K x S
           x = x.permute(0, 3, 1, 2).contiguous()
        if x.dim() == 3:
            x = torch.transpose(x, 1, 2)
            # N x L x C == only channel norm
            x = super().forward(x)
            # N x C x L
            x = torch.transpose(x, 1, 2)
        return x


def select_norm(norm, dim, shape):
    if norm == 'gln':
        return GlobalLayerNorm(dim, shape, elementwise_affine=True)
    if norm == 'cln':
        return CumulativeLayerNorm(dim, elementwise_affine=True)
    if norm == 'ln':
        return nn.GroupNorm(1, dim, eps=1e-8)
    else:
        return nn.BatchNorm2d(dim)




class Dual_RNN_Block(nn.Module):
    '''
       Implementation of the intra-RNN and the inter-RNN
       input:
            in_channels: The number of expected features in the input x
            out_channels: The number of features in the hidden state h
            rnn_type: RNN, LSTM, GRU
            norm: gln = "Global Norm", cln = "Cumulative Norm", ln = "Layer Norm"
            dropout: If non-zero, introduces a Dropout layer on the outputs 
                     of each LSTM layer except the last layer, 
                     with dropout probability equal to dropout. Default: 0
            bidirectional: If True, becomes a bidirectional LSTM. Default: False
    '''

    def __init__(self, out_channels,
                 hidden_channels, rnn_type='LSTM', norm='ln',
                 dropout=0, bidirectional=False):
        super(Dual_RNN_Block, self).__init__()
        # RNN model
        self.intra_rnn = getattr(nn, rnn_type)(
            out_channels, hidden_channels, 1, batch_first=True, dropout=dropout, bidirectional=False)
        self.inter_rnn = getattr(nn, rnn_type)(
            out_channels, hidden_channels, 1, batch_first=True, dropout=dropout, bidirectional=False)
        # Norm
        self.intra_norm = select_norm(norm, out_channels, 4)
        self.inter_norm = select_norm(norm, out_channels, 4)
        # Linear
        self.intra_linear = nn.Linear(
            hidden_channels*2 if bidirectional else hidden_channels, out_channels)
        self.inter_linear = nn.Linear(
            hidden_channels*2 if bidirectional else hidden_channels, out_channels) # if bidirectional else hidden_channels
        

    def forward(self, x):
        '''
           x: [B, N, K, S]
           out: [Spks, B, N, K, S]
        '''
        B, N, K, S = x.shape
        # intra RNN
        # [BS, K, N]
        intra_rnn = x.permute(0, 3, 2, 1).contiguous().view(B*S, K, N)
        # [BS, K, H]
        intra_rnn, _ = self.intra_rnn(intra_rnn)
        # [BS, K, N]
        intra_rnn = self.intra_linear(intra_rnn.contiguous().view(B*S*K, -1)).view(B*S, K, -1)
        # [B, S, K, N]
        intra_rnn = intra_rnn.view(B, S, K, N)
        # [B, N, K, S]
        intra_rnn = intra_rnn.permute(0, 3, 2, 1).contiguous()
        intra_rnn = self.intra_norm(intra_rnn)
        
        # [B, N, K, S]
        intra_rnn = intra_rnn + x

        # inter RNN
        # [BK, S, N]
        inter_rnn = intra_rnn.permute(0, 2, 3, 1).contiguous().view(B*K, S, N)
        # [BK, S, H]
        inter_rnn, state = self.inter_rnn(inter_rnn)
        # [BK, S, N]
        inter_rnn = self.inter_linear(inter_rnn.contiguous().view(B*S*K, -1)).view(B*K, S, -1)
        # [B, K, S, N]
        inter_rnn = inter_rnn.view(B, K, S, N)
        # [B, N, K, S]
        inter_rnn = inter_rnn.permute(0, 3, 1, 2).contiguous()
        inter_rnn = self.inter_norm(inter_rnn)
        # [B, N, K, S]
        out = inter_rnn + intra_rnn

        return out


class Dual_Path_RNN(nn.Module):
    '''
       Implementation of the Dual-Path-RNN model
       input:
            in_channels: The number of expected features in the input x
            out_channels: The number of features in the hidden state h
            rnn_type: RNN, LSTM, GRU
            norm: gln = "Global Norm", cln = "Cumulative Norm", ln = "Layer Norm"
            dropout: If non-zero, introduces a Dropout layer on the outputs 
                     of each LSTM layer except the last layer, 
                     with dropout probability equal to dropout. Default: 0
            bidirectional: If True, becomes a bidirectional LSTM. Default: False
            num_layers: number of Dual-Path-Block
            K: the length of chunk
            num_spks: the number of speakers
    '''

    def __init__(self, in_channels, out_channels, hidden_channels,
                 rnn_type='GRU', norm='ln', dropout=0,
                 bidirectional=False,  K=200):
        super(Dual_Path_RNN, self).__init__()
        self.K = K
        # self.conv_block1 = nn.Sequential(
            # nn.Conv1d(in_channels, 64, kernel_size=8,
                      # stride=1, bias=False, padding=4),
            # nn.BatchNorm1d(64),
            # nn.ReLU(),
            # nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            # nn.Dropout(0.35)
        # )#####8   1    4
        
        #self.dual_rnn = nn.ModuleList([])
        #self.MaxPoolinglayer = nn.MaxPool1d(kernel_size=4, stride=4, padding=2)
        
        self.layer1 =  Dual_RNN_Block(in_channels, hidden_channels, rnn_type=rnn_type, norm=norm, dropout=dropout, bidirectional=bidirectional)
        self.layer2 =  Dual_RNN_Block(in_channels, hidden_channels, rnn_type=rnn_type, norm=norm, dropout=dropout, bidirectional=bidirectional)
        self.layer3 =  Dual_RNN_Block(in_channels, hidden_channels, rnn_type=rnn_type, norm=norm, dropout=dropout, bidirectional=bidirectional)
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1,stride=1, bias=False, padding=0),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            #nn.MaxPool1d(kernel_size=2, stride=2, padding=1),# 4,4,2
        )

        # self.conv_block3 = nn.Sequential(
            # nn.Conv1d(39, out_channels, kernel_size=1,stride=1, bias=False, padding=0),
            # #nn.BatchNorm1d(out_channels),
            # nn.ReLU(),
            # # nn.MaxPool1d(kernel_size=3, stride=2, padding=1),# 4,4,2
        # )

    def forward(self, x):
        '''
           x: [B, N, L]
        '''
        # [B, N, L]
        #x = self.conv_block1(x)
        #print(x.size())
        # [B, N, K, S]
        x, gap = self._Segmentation(x, self.K)
        #print(x.size())
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        #print(x.size())
        x = self._over_add(x, gap)
        
        x1 = self.conv_block2(x)
        #x2 = self.conv_block3(x.permute(0,2,1))
        #print(x.size())
        # x = self.conv_block3(x)
        #print(x.size())

        return x1

    def _padding(self, input, K):
        '''
           padding the audio times
           K: chunks of length
           P: hop size
           input: [B, N, L]
        '''
        B, N, L = input.shape
        P = K // 2
        gap = K - (P + L % K) % K
        if gap > 0:
            pad = torch.Tensor(torch.zeros(B, N, gap)).type(input.type())
            input = torch.cat([input, pad], dim=2)

        _pad = torch.Tensor(torch.zeros(B, N, P)).type(input.type())
        input = torch.cat([_pad, input, _pad], dim=2)

        return input, gap

    def _Segmentation(self, input, K):
        '''
           the segmentation stage splits
           K: chunks of length
           P: hop size
           input: [B, N, L]
           output: [B, N, K, S]
        '''
        B, N, L = input.shape
        P = K // 2
        input, gap = self._padding(input, K)
        # [B, N, K, S]
        input1 = input[:, :, :-P].contiguous().view(B, N, -1, K)
        input2 = input[:, :, P:].contiguous().view(B, N, -1, K)
        input = torch.cat([input1, input2], dim=3).view(
            B, N, -1, K).transpose(2, 3)

        return input.contiguous(), gap

    def _over_add(self, input, gap):
        '''
           Merge sequence
           input: [B, N, K, S]
           gap: padding length
           output: [B, N, L]
        '''
        B, N, K, S = input.shape
        P = K // 2
        # [B, N, S, K]
        input = input.transpose(2, 3).contiguous().view(B, N, -1, K * 2)

        input1 = input[:, :, :, :K].contiguous().view(B, N, -1)[:, :, P:]
        input2 = input[:, :, :, K:].contiguous().view(B, N, -1)[:, :, :-P]
        input = input1 + input2
        # [B, N, L]
        if gap > 0:
            input = input[:, :, :-gap]

        return input

class TFRNN(nn.Module):
    def __init__(self, class_number):
        super(TFRNN, self).__init__()
        
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=33,
                      stride=8, bias=False, padding=(33//2)),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.1)
        )
        
        # self.conv_block2 = nn.Sequential(
            # nn.Conv1d(32, 32, kernel_size=65,
                      # stride=16, bias=False, padding=(65//2)),
            # nn.BatchNorm1d(32),
            # nn.ReLU(),
            # nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            # nn.Dropout(0.1)
        # )        
        
        self.DPR3 = Dual_Path_RNN(32, 128, 32,
         rnn_type='GRU', norm='ln', dropout=0.2,
         bidirectional=False,  K=32) ####configs.input_channels, configs.final_out_channels
        self.avgpool = nn.AdaptiveAvgPool1d((1))
        self.logits = nn.Linear(128, class_number)
    def forward(self, x_in):
        #print(x_in.size())
        x = self.conv_block1(x_in)
        # x = self.conv_block2(x)
        x = self.DPR3(x)  
        #print(x.size())
        #x = x.reshape(x.shape[0],x.shape[1],-1)
        x = self.avgpool(x).reshape(x.shape[0], -1)
        x_flat = self.logits(x)
        #print(x_flat.shape)
        return x_flat, x  #logits   

class GCN(nn.Module):
    def  __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.dropout = dropout
        self.avgpool = nn.AdaptiveAvgPool1d((1))
        self.logits = nn.Linear(nhid, nclass)

    def forward(self, x, adj):    #x特征矩阵,agj邻接矩阵 
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.avgpool(self.gc2(x, adj)).reshape(x.shape[0], -1)
        #x = self.gc2(x, adj)
        #print(x.shape,"aaaaaaaaaa")
        x_flat = self.logits(x)
         
        return x_flat, x

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Conv1d(in_features, out_features, kernel_size=3, padding=1)##nn.Parameter(torch.FloatTensor(in_features, out_features))

    def forward(self, input, adj):
        #print(input.shape)
        support = self.weight(input)
        #print(support.shape)
        output = torch.mm(support.reshape(-1,20),adj)
        output = output.reshape(-1,self.out_features,20)
        #print(output.shape)
        
        return output


class Mut_Modal(nn.Module):
    def __init__(self, class_number):
        super(Mut_Modal, self).__init__()
        
        #self.series = GCN(20,20,class_number,0.1)#DPRNN(class_number)
        self.series = GCN(1,128,class_number,0.1)
        self.text = TFRNN(class_number)
        self.logits = nn.Linear(256, class_number)
    def forward(self, x, x1,x2):
        #print(x_in.size())
        x_cls,x_feat = self.series(x,x2)
        x1_cls,x1_feat = self.text(x1)
        #print(x.size())
        #x = x.reshape(x.shape[0],x.shape[1],-1)

        all_cls = self.logits(torch.cat([x_feat, x1_feat],1))
        #print(x_flat.shape)
        return    all_cls, x_cls, x1_cls



class Mut_Modal_text(nn.Module):
    def __init__(self, class_number):
        super(Mut_Modal_text, self).__init__()
        
        #self.series = DPRNN(class_number)#DPRNN(class_number)
        self.text = TFRNN(class_number)
        self.logits = nn.Linear(128, class_number)
    def forward(self, x1):
        #print(x_in.size())
        #x_cls,x_feat = self.series(x)
        x1_cls,x1_feat = self.text(x1)
        #print(x.size())
        #x = x.reshape(x.shape[0],x.shape[1],-1)

        #all_cls = self.logits(x1_feat)
        #print(x_flat.shape)
        return    x1_cls, x1_cls, x1_cls  
  

class Mut_Modal_series(nn.Module):
    def __init__(self, class_number):
        super(Mut_Modal_series, self).__init__()
        
        self.series = DPRNN(class_number)#DPRNN(class_number)
        #self.text = TFRNN(class_number)
        self.logits = nn.Linear(128, class_number)
    def forward(self, x):
        #print(x_in.size())
        x_cls,x_feat = self.series(x)
        #x1_cls,x1_feat = self.text(x1)
        #print(x.size())
        #x = x.reshape(x.shape[0],x.shape[1],-1)

        #all_cls = self.logits(x1_feat)
        #print(x_flat.shape)
        return    x_cls, x_cls, x_cls  
