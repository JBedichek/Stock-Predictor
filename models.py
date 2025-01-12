import torch
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaModel
import math
import yfinance as yf
from math import sqrt

from torch import Tensor
from typing import Optional, Any, Union, Callable
import torch.nn.functional as F
import random

class RobertaEncoder(nn.Module):
    def __init__(self, device='cuda:0'):
        super().__init__()
        self.device = device
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base') 
        self.model = RobertaModel.from_pretrained('roberta-base')

    def forward(self, text):
        e = self.tokenizer(text, return_tensors='pt').to(self.device)
        output = self.model(**e)
        return output.last_hidden_state[:,0,:]
 
class c_transformer_layer(nn.Module):
    def __init__(self, static_dim, seq_dim, act_fn, data_dim, nhead, dim_ff, dropout):
        super(c_transformer_layer, self).__init__()
        self.static_dim = static_dim
        self.seq_dim = seq_dim
        self.data_dim = data_dim
        self.nhead = nhead
        self.dim_ff = dim_ff
        self.act_fn = act_fn
        self.seq_len = int(self.seq_dim/data_dim)
        self.lin_inp = nn.Sequential(
            nn.Linear(self.static_dim+self.seq_dim, self.seq_dim),
            self.act_fn(),
            #nn.Linear(self.seq_dim, self.seq_dim),
            #self.act_fn()
        )
        self.tran_layer = nn.TransformerEncoderLayer(d_model=self.data_dim, nhead=nhead, dim_feedforward=dim_ff, 
                                    activation=self.act_fn(), batch_first = True, dropout=dropout)
        
        
    def forward(self, x, sum):
        batch_size = x.shape[0]
        x = self.tran_layer(x)
        res = torch.reshape(x, (batch_size, self.seq_dim))
        x = torch.cat((torch.reshape(x, (batch_size, self.seq_dim)), sum), dim=1)
        x = self.lin_inp(x)
        x = x + res
        x = torch.reshape(x, (batch_size, self.seq_len, self.data_dim))
        return x

class _base_transformer_layer(nn.Module):
    def __init__(self, act_fn, data_dim, nhead, dim_ff, dropout):
        super(_base_transformer_layer, self).__init__()
        self.attn_dim = data_dim*3
        self.attn = nn.MultiheadAttention(data_dim, nhead, dropout,kdim=self.attn_dim, vdim=self.attn_dim,
                                          batch_first=True)
        self.tran_layer = nn.TransformerEncoderLayer(d_model=data_dim, nhead=nhead, dim_feedforward=dim_ff, 
                                    activation=act_fn(), batch_first=True, dropout=dropout)
        
    def forward(self, x):
        x = self.tran_layer(x)
        return x
     
class base_transformer_layer(nn.Module):
    def __init__(self, act_fn, data_dim, nhead, dim_ff, dropout):
        super(base_transformer_layer, self).__init__()
        self.tran_layer = nn.TransformerEncoderLayer(d_model=data_dim, nhead=nhead, dim_feedforward=dim_ff, 
                                    activation=act_fn(), batch_first=True, dropout=dropout, norm_first=True)
        
    def forward(self, x):
        x = self.tran_layer(x)
        return x

class Dist_Pred(nn.Module):
    def __init__(self,seq_len=350, data_dim=5, num_bins=21, num_days=5, nhead=5, ff=15000, layers=72, sum_emb=76, scale=1, s_scale=0, num_cls_layers=6, dropout=0.1):
        super(Dist_Pred, self).__init__()
        self.num_bins = num_bins
        self.seq_len = seq_len
        self.dim = data_dim
        self.num_preds = num_days-1
        self.act_fn = nn.GELU
        self.act = nn.GELU()
        self.scale = scale
        self.s_scale = s_scale
        self.sum_emb = sum_emb
        self.seq_dim = self.seq_len*self.dim
        self.num_lin_layers = num_cls_layers
        self.dropout = nn.Dropout(dropout)
        # Transformer Layers
        self.layers = nn.ModuleList([base_transformer_layer(act_fn=self.act_fn,data_dim=self.dim, nhead=nhead, 
                                            dim_ff=ff, dropout=0.1) for i in range(layers)])  
        linear_in_dim = 700
        # Classification Head
        '''
        self.linear_layers = nn.ModuleList([nn.Linear(int(self.scale*self.seq_len*data_dim), int(self.scale*self.seq_len*data_dim)) for i in range(self.num_lin_layers)]) 
        self.linear_out = nn.Linear(int(self.scale*self.seq_len*data_dim), num_bins*self.num_preds)
        self.linear_in = nn.Sequential(
            nn.Linear(self.seq_len*self.dim+self.sum_emb, int(self.scale*self.seq_len*data_dim)),
            self.act_fn(),)
        '''
        self.linear_in = nn.Sequential(
            nn.Linear(self.seq_len*self.dim+self.sum_emb+156, linear_in_dim),
            self.act_fn(),
            nn.Linear(linear_in_dim, linear_in_dim*3),
            self.act_fn(),
            nn.Linear(linear_in_dim*3, linear_in_dim*3),
            self.act_fn(),
            nn.Linear(linear_in_dim*3, num_bins*self.num_preds),
            )

        print('Linear Params: ', sum(param.numel() for param in self.linear_in.parameters()))
        print('Transformer params ', sum(param.numel() for param in self.layers.parameters()))
        
        
        # Summary Module
        #self.summary_in = nn.Sequential(nn.Linear(sum_emb, int(self.s_scale*sum_emb)),
        #                                self.act_fn())
        #self.summary_layers = nn.ModuleList([nn.Linear(int(self.s_scale*sum_emb), int(self.s_scale*sum_emb))])
        self.pos_encoding = PositionalEncoding(data_dim, seq_len)
        self._encoding = nn.Parameter(self.pos_encoding.encoding, requires_grad=False)

    # For use in forward()
    def pos_encode(self, x):
        batch_size, seq_len, data_dim = x.size()
        return self._encoding[:seq_len, :]
    
    def forward(self, x, s):
        batch_size = x.shape[0]
        x = torch.flip(x,[1])
        x = x + self.pos_encode(x)
        
        # Reshape this to (batch, _, 52) so it can be appended to the end of the sequence
        s = torch.reshape(s, (batch_size, 19, 52))
        
        # Add these data points to existing seqence
        x = torch.cat((x, s), dim=1)

        # Send the final data through transformer layers
        for layer in self.layers:
            x = layer(x)

        # Send transformer activation through linear classification head
        x = torch.reshape(x, (batch_size, self.seq_dim+self.sum_emb+156))

        x = self.linear_in(x)
        #for lin_layer in self.linear_layers:
        #    x = lin_layer(x)
        #    x = self.act(x)
            # x = self.dropout(x)
        #x = self.linear_out(x)

        # Return reshaped output
        x = torch.reshape(x, (batch_size, self.num_bins, self.num_preds))
        return x
    
    def transformer(self, x, s):
        '''
        Returns the transformer activation of the network
        for downstream greedy training.
        '''
        
        batch_size = x.shape[0]
        x = torch.flip(x, [1])
        x = x + self.pos_encode(x)
        s = torch.reshape(s, (batch_size, 19, 52))
        x = torch.cat((x, s), dim=1)
        for layer in self.layers:
            x = layer(x)

        return x

def temp_softmax(tensor, temp=1.0):
    softmax = torch.nn.Softmax(dim=1)
    return softmax(tensor/temp)

class t_Dist_Pred(nn.Module):
    def __init__(self,seq_len=350, data_dim=5, num_bins=21, num_days=5, nhead=5, ff=15000, layers=72, sum_emb=76, scale=1, s_scale=0, num_cls_layers=6, dropout=0.15):
        super(t_Dist_Pred, self).__init__()
        self.num_bins = num_bins
        self.seq_len = seq_len
        self.dim = data_dim
        self.num_preds = num_days-1
        self.act_fn = nn.GELU
        self.act = nn.GELU()
        self.scale = scale
        self.s_scale = s_scale
        self.sum_emb = sum_emb
        self.seq_dim = self.seq_len*self.dim
        self.num_lin_layers = num_cls_layers
        self.softmax = temp_softmax
        self.dropout = nn.Dropout(dropout)
        # Transformer Layers
        self.layers = nn.ModuleList([base_transformer_layer(act_fn=self.act_fn,data_dim=self.dim, nhead=nhead, 
                                            dim_ff=ff, dropout=0.1) for i in range(layers)])  
        
        linear_in_dim = 2200
        # Classification Head
        '''
        self.linear_layers = nn.ModuleList([nn.Linear(int(self.scale*self.seq_len*data_dim), int(self.scale*self.seq_len*data_dim)) for i in range(self.num_lin_layers)]) 
        self.linear_out = nn.Linear(int(self.scale*self.seq_len*data_dim), num_bins*self.num_preds)
        self.linear_in = nn.Sequential(
            nn.Linear(self.seq_len*self.dim+self.sum_emb, int(self.scale*self.seq_len*data_dim)),
            self.act_fn(),)
        '''
        self.linear_in = nn.Sequential(
            self.dropout,
            #nn.LayerNorm(normalized_shape=(1208)),
            nn.Linear(604*2, linear_in_dim),
            self.act_fn(),
            self.dropout,
            nn.Linear(linear_in_dim, linear_in_dim),
            self.act_fn(),
            self.dropout,
            nn.Linear(linear_in_dim, num_bins*self.num_preds)
            )

        print('Linear Params: ', sum(param.numel() for param in self.linear_in.parameters()))
        print('Transformer params ', sum(param.numel() for param in self.layers.parameters()))
        
        
        # Summary Module
        #self.summary_in = nn.Sequential(nn.Linear(sum_emb, int(self.s_scale*sum_emb)),
        #                                self.act_fn())
        #self.summary_layers = nn.ModuleList([nn.Linear(int(self.s_scale*sum_emb), int(self.s_scale*sum_emb))])
        self.pos_emb = nn.Embedding(seq_len,data_dim)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)

        self.pos_encoding = PositionalEncoding(data_dim, seq_len)
        self._encoding = nn.Parameter(self.pos_encoding.encoding, requires_grad=False)
        self.stochastic_depth_prob = 0.4
        self.layer_drop_probs = [((i+1)/layers)*self.stochastic_depth_prob for i in range(layers)]

    # For use in forward()
    def pos_encode(self, x):
        batch_size, seq_len, data_dim = x.size()
        return self._encoding[:seq_len, :]
    
    def forward(self, x, s):
        #print(x.shape)
        batch_size = x.shape[0]
        #x = x[:,:,:243]
        #print(x.shape, s.shape)
        #x = torch.flip(x,[1])
        #x = x + self.pos_encode(x)
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        pos_emb = self.pos_emb(positions)
        pos_emb = self.dropout(pos_emb)
        x = x + pos_emb
        
        # Reshape this to (batch, _, 52) so it can be appended to the end of the sequence
        #print(s.shape)
        s = s[:,:218*4]
        #print(s.shape)
        s = torch.reshape(s, (batch_size, 4, 218))
        
        # Add these data points to existing seqence
        x = torch.cat((x, s), dim=1)

        # Send the final data through transformer layers
        init_res1 = x
        init_res2 = 0
        init_res3 = 0
        init_res4 = 0
        init_res5 = 0
        #i = 0
        for i, layer in enumerate(self.layers):
            #x = layer(x) + init_res1*0.6+init_res2*0.6+init_res3*0.6+init_res4*0.6+init_res5*0.6
            if random.random() > self.layer_drop_probs[i] and self.training:
                x = layer(x) + init_res1
            elif self.training:
                x = x
            else:
                x = layer(x)*(1-self.layer_drop_probs[i]) + init_res1
            #x = layer(x)
            #if i == 2:
            #    init_res2 = x
            #if i == 4:
            #    init_res3 = x
            #if i == 6:
            #    init_res4 = x
            #if i == 8:
            #    init_res5 = x
            if i % int(len(self.layers)/7) == 0: 
                init_res1 = x
            #i += 1
        # Send transformer activation through linear classification head
        #print(x.shape)
        x1 = torch.mean(x[:,:,int(x.shape[2]/2):], dim=2)
        x2 = torch.mean(x[:,:,:int(x.shape[2]/2)], dim=2)
        #print(x1.shape, x2.shape)
        x = torch.cat((x1, x2), dim=1)
        #x = x.squeeze(2)
        #x = torch.reshape(x, (batch_size, self.seq_dim+self.sum_emb+156))
        #x = torch.reshape(x, (batch_size, 619*2))

        x = self.linear_in(x)
        #for lin_layer in self.linear_layers:
        #    x = lin_layer(x)
        #    x = self.act(x)
            # x = self.dropout(x)
        #x = self.linear_out(x)

        # Return reshaped output
        x = torch.reshape(x, (batch_size, self.num_bins, self.num_preds))
        # = self.softmax(x)
        if not self.training:
            x = self.softmax(x)
        return x
    
    def forward_with_t_act(self, x, s):
        batch_size = x.shape[0]
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        pos_emb = self.pos_emb(positions)
        pos_emb = self.dropout(pos_emb)
        x = x + pos_emb
        
        # Reshape this to (batch, _, 52) so it can be appended to the end of the sequence
        s = s[:,:218*4]
        s = torch.reshape(s, (batch_size, 4, 218))
        
        # Add these data points to existing seqence
        x = torch.cat((x, s), dim=1)

        # Send the final data through transformer layers
        init_res1 = x
        for i, layer in enumerate(self.layers):
            x = layer(x) + init_res1
            if i % int(len(self.layers)/4) == 0: 
                init_res1 = x
            #i += 1
        # Send transformer activation through linear classification head
        #print(x.shape)
        x1 = torch.mean(x[:,:,int(x.shape[2]/2):], dim=2)
        x2 = torch.mean(x[:,:,:int(x.shape[2]/2)], dim=2)
        #print(x1.shape, x2.shape)
        t_act = torch.cat((x1, x2), dim=1)


        x = self.linear_in(t_act)

        # Return reshaped output
        x = torch.reshape(x, (batch_size, self.num_bins, self.num_preds))
        x = self.softmax(x)

        return x, t_act
    
    def transformer(self, x, s):
        '''
        Returns the transformer activation of the network
        for downstream greedy training.
        '''
        
        batch_size = x.shape[0]
        x = torch.flip(x, [1])
        x = x + self.pos_encode(x)
        s = torch.reshape(s, (batch_size, 19, 52))
        x = torch.cat((x, s), dim=1)
        for layer in self.layers:
            x = layer(x)

        return x

class L1_Dist_Pred(nn.Module):
    def __init__(self,seq_len=350, data_dim=5, num_bins=21, num_days=5, nhead=5, ff=15000, layers=72, sum_emb=76, scale=1, s_scale=0, dropout=0.1):
        super(L1_Dist_Pred, self).__init__()
        self.num_bins = num_bins
        self.seq_len = seq_len
        self.dim = data_dim
        self.num_preds = num_days-1
        self.act_fn = nn.GELU
        self.act = nn.GELU()
        self.scale = scale
        self.s_scale = s_scale
        self.sum_emb = 768
        self.seq_dim = (self.seq_len+19)*self.dim
        self.dropout = nn.Dropout(dropout)
        
        # Transformer Layers
        self.layers = nn.ModuleList([base_transformer_layer(act_fn=self.act_fn,data_dim=self.dim, nhead=nhead, 
                                            dim_ff=ff, dropout=0.1) for i in range(layers)])  
        linear_in_dim = 800
        
        # Classification Head
        self.linear_in = nn.Sequential(
            nn.Linear(self.seq_dim, linear_in_dim),
            self.act_fn())
        
        self.cls_head_in = nn.Sequential(
            nn.Linear(320*4, 320*4+200),
            self.act_fn(),
            nn.Linear(320*4+200, 320*4+200),
            self.act_fn())

        self.linear_out = nn.Sequential(
            nn.Linear(linear_in_dim+320*4+200, int(linear_in_dim*2.5)),
            self.act_fn(),
            nn.Linear(int(linear_in_dim*2.5), int(linear_in_dim*2.5)),
            self.act_fn(),
            nn.Linear(int(linear_in_dim*2.5), num_bins*self.num_preds))

        print('Linear Params: ', sum(param.numel() for param in self.linear_in.parameters()))
        print('Transformer params ', sum(param.numel() for param in self.layers.parameters()))
    
    def forward(self, x, pred):
        batch_size = x.shape[0]
        #print(x.shape)
        # Send the data through transformer layers
        for layer in self.layers:
            x = layer(x)

        # Send transformer activation through linear classification head
        x = torch.reshape(x, (batch_size, self.seq_dim))
        
        pred = self.cls_head_in(torch.reshape(pred, (batch_size, self.num_bins*4)))
        x = self.linear_in(x)
        x = torch.cat((x, pred), dim=1)
        x = self.linear_out(x)

        # Return reshaped output
        x = torch.reshape(x, (batch_size, self.num_bins, self.num_preds))
        return x

    def transformer(self, x):
        batch_size = x.shape[0]
        #print(x.shape)
        # Send the data through transformer layers
        for layer in self.layers:
            x = layer(x)
        return x

class L2_Dist_Pred(nn.Module):
    def __init__(self,seq_len=350, data_dim=5, num_bins=21, num_days=5, nhead=5, ff=15000, layers=72, sum_emb=76, scale=1, s_scale=0, dropout=0.1):
        super(L2_Dist_Pred, self).__init__()
        self.num_bins = num_bins
        self.seq_len = seq_len
        self.dim = data_dim
        self.num_preds = num_days-1
        self.act_fn = nn.GELU
        self.act = nn.GELU()
        self.scale = scale
        self.s_scale = s_scale
        self.sum_emb = 768
        self.seq_dim = (self.seq_len+19)*self.dim
        self.dropout = nn.Dropout(dropout)
        
        # Transformer Layers
        self.layers = nn.ModuleList([base_transformer_layer(act_fn=self.act_fn,data_dim=self.dim, nhead=nhead, 
                                            dim_ff=ff, dropout=0.1) for i in range(layers)])  
        linear_in_dim = 700
        
        # Classification Head
        self.linear_in = nn.Sequential(
            nn.Linear(self.seq_dim, linear_in_dim),
            self.dropout,
            self.act_fn())
        
        self.cls_head_in = nn.Sequential(
            nn.Linear(320*4, 320*4+200),
            self.act_fn(),
            nn.Linear(320*4+200, 320*4+200),
            self.dropout,
            self.act_fn())

        self.linear_out = nn.Sequential(
            nn.Linear(linear_in_dim+320*4+200, int(linear_in_dim*2.5)),
            self.dropout,
            self.act_fn(),
            nn.Linear(int(linear_in_dim*2.5), int(linear_in_dim*2.5)),
            self.act_fn(),
            nn.Linear(int(linear_in_dim*2.5), num_bins*self.num_preds))

        print('Linear Params: ', sum(param.numel() for param in self.linear_in.parameters()))
        print('Transformer params ', sum(param.numel() for param in self.layers.parameters()))
    
    def forward(self, x, pred):
        batch_size = x.shape[0]
        #print(x.shape)
        # Send the data through transformer layers
        for layer in self.layers:
            x = layer(x)

        # Send transformer activation through linear classification head
        x = torch.reshape(x, (batch_size, self.seq_dim))
        
        pred = self.cls_head_in(torch.reshape(pred, (batch_size, self.num_bins*4)))
        x = self.linear_in(x)
        x = torch.cat((x, pred), dim=1)
        x = self.linear_out(x)

        # Return reshaped output
        x = torch.reshape(x, (batch_size, self.num_bins, self.num_preds))
        return x

class Full_L1_Dist_Pred(nn.Module):
    def __init__(self, base_pth, layer_pth, train=True):
        super(Full_L1_Dist_Pred, self).__init__()
        self.base = torch.load(base_pth)
        self.layer = torch.load(layer_pth)
        if not train:
            for param in self.base.parameters():
                param.requires_grad = False
            for param in self.layer.parameters():
                param.requires_grad = False
        
    
    def forward(self, data, sum):
        batch_size = data.shape[0]
        pred = self.base(data, sum)
        t_act = self.base.transformer(data, sum)
        data = torch.flip(data,[1])
        data = data + self.base.pos_encode(data)
        sum = torch.reshape(sum, (batch_size, 19, 52))
        data = torch.cat((data, sum), dim=1)
        t_act = torch.cat((t_act, data), dim=2)
        layer_pred = self.layer(t_act, pred)
        return layer_pred


class Composed_Dist_Pred(nn.Module):
    def __init__(self, base_pth, layer_pth, layer2_pth, train=False):
        super(Composed_Dist_Pred, self).__init__()
        self.base = torch.load(base_pth)
        self.layer = torch.load(layer_pth)
        self.layer2 = torch.load(layer2_pth)
        if not train:
            for param in self.base.parameters():
                param.requires_grad = False
            for param in self.layer.parameters():
                param.requires_grad = False
            for param in self.layer2.parameters():
                param.requires_grad = False
        
        
    
    def forward(self, data, sum):
        batch_size = data.shape[0]
        pred = self.base(data, sum)
        t_act = self.base.transformer(data, sum)
        data = torch.flip(data,[1])
        data = data + self.base.pos_encode(data)
        sum = torch.reshape(sum, (batch_size, 19, 52))
        data = torch.cat((data, sum), dim=1)
        t_act = torch.cat((t_act, data), dim=2)
        layer_pred = self.layer(t_act, pred)
        layer_t_act = self.layer.transformer(t_act)
        t_activation = torch.cat((data, layer_t_act), dim=2)
        x = self.layer2(t_activation, layer_pred)
        softmax = nn.Softmax(dim=1)
        return softmax(x)

class Full_Dist_Pred(nn.Module):
    def __init__(self, base_pth):
        super(Full_Dist_Pred, self).__init__()
        self.base = torch.load(base_pth)
        self.extension = nn.Sequential(
            nn.Linear(4*320, 2200),
            nn.BatchNorm1d(2200),
            nn.GELU(),
            nn.Linear(2200, 2200),
            nn.BatchNorm1d(2200),
            nn.GELU(),
            nn.Linear(2200, 2200),
            nn.BatchNorm1d(2200),
            nn.GELU(),
            nn.Linear(2200, 2200),
            nn.BatchNorm1d(2200),
            nn.GELU(),
            nn.Linear(2200, 320*4)
        )
        
    
    def forward(self, data, sum):
        batch_size = data.shape[0]
        _pred = self.base(data, sum)
        _pred = torch.reshape(_pred, (batch_size, 320*4))
        pred = self.extension(_pred)
        pred = pred + _pred
        return torch.reshape(pred, (batch_size, 320, 4))

# The following is from hyunwoongko's implementation of the Transformer model: https://github.com/hyunwoongko/transformer
class PositionalEncoding(nn.Module):
    """
    compute sinusoid encoding.
    """
    def __init__(self, d_model, max_len):
        """
        constructor of sinusoid encoding class

        :param d_model: dimension of model
        :param max_len: max sequence length
        :param device: hardware device setting
        """
        super(PositionalEncoding, self).__init__()

        # same size with input matrix (for adding with input matrix)
        self.encoding = torch.zeros(max_len, d_model)
        self.encoding.requires_grad = False  # we don't need to compute gradient
        pos = torch.arange(0, max_len)
        pos = pos.float().unsqueeze(dim=1)
        # 1D => 2D unsqueeze to represent word's position

        _2i = torch.arange(0, d_model, step=2).float()
        # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        if d_model % 2 == 0:
            self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        else:
            self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i[:d_model//2] / d_model)))
        # compute positional encoding to consider positional information of words

    def forward(self, x):
        # self.encoding
        # [max_len = 512, d_model = 512]
        #print(x.size())
        batch_size, seq_len, data_dim = x.size()
        # [batch_size = 128, seq_len = 30]

        return self.encoding[:seq_len, :]
        # [seq_len = 30, d_model = 512]
        # it will add with tok_emb : [128, 30, 512]         

class meta_model(nn.Module):
    def __init__(self, width, depth):
        super(meta_model, self).__init__()
        self.input = nn.Linear(2408, width)
        self.network = nn.ModuleList([nn.Linear(width,width) for i in range(depth)])
        self.output = nn.Linear(width,1)
        self.act_fn = nn.GELU()
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, t_act, pred):
        batch_size = pred.shape[0]
        pred = torch.reshape(pred, (batch_size, 300*4))
        t_act = t_act.squeeze(1)
        #print(t_act.shape)
        x = torch.cat((t_act, pred), dim=1)
    
        
        x = self.input(x)
        x = self.act_fn(x)
        for layer in self.network:
            x = layer(x)
            x = self.act_fn(x)
            x = self.dropout(x)
        #softmax = nn.Softmax(dim=1)
        return self.output(x)

   