# Copyright (c) Hefei University of Technology. 

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import math
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from collections import OrderedDict
import copy
from typing import Optional, List
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from classification.models.vmamba import VSSBlock_gather_4
from einops import rearrange, repeat



BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)
    
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def index_reverse(index):
    index_r = torch.zeros_like(index)
    ind = torch.arange(0, index.shape[-1]).to(index.device)
    for i in range(index.shape[0]):
        index_r[i, index[i, :]] = ind
    return index_r

def semantic_neighbor(x, index):#X:B,HW,hidden ； index：B,HW
    dim = index.dim() 
    assert x.shape[:dim] == index.shape, "x ({:}) and index ({:}) shape incompatible".format(x.shape, index.shape)

    for _ in range(x.dim() - index.dim()): 
        index = index.unsqueeze(-1)
    index = index.expand(x.shape) 

    shuffled_x = torch.gather(x, dim=dim - 1, index=index)
    return shuffled_x

class dwconv(nn.Module):
    def __init__(self, hidden_features, kernel_size=5):
        super(dwconv, self).__init__()
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=kernel_size, stride=1,
                      padding=(kernel_size - 1) // 2, dilation=1,
                      groups=hidden_features), nn.GELU())
        self.hidden_features = hidden_features

    def forward(self, x, x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.hidden_features, x_size[0], x_size[1]).contiguous()  # b Ph*Pw c
        x = self.depthwise_conv(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x

class ConvFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, kernel_size=5, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.dwconv = dwconv(hidden_features=hidden_features, kernel_size=kernel_size)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x, x_size):
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.dwconv(x, x_size)
        x = self.fc2(x)
        return x
    
class Selective_Scan(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=1, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=1, merge=True)  # (K=4, D, N)
        self.selective_scan = selective_scan_fn

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor, prompt,gather_type):
        B, L, C = x.shape
        K = 1  # mambairV2 needs noly 1 scan
        xs = x.permute(0, 2, 1).view(B, 1, C, L).contiguous()  # B, 1, C ,L

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        #  our ASE here ---
        if gather_type==0:
            Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        else:
            Cs = Cs.float().view(B, K, -1, L) + prompt  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        return out_y[:, 0]

    def forward(self, x: torch.Tensor, prompt,gather_type, **kwargs):
        b, l, c = prompt.shape
        prompt = prompt.permute(0, 2, 1).contiguous().view(b, 1, c, l)
        y = self.forward_core(x, prompt,gather_type)  # [B, L, C]
        y = y.permute(0, 2, 1).contiguous()
        return y

class ASSM(nn.Module):
    def __init__(self, dim, d_state, num_tokens=17, inner_rank=128, mlp_ratio=2.):
        super().__init__()
        self.dim = dim
        self.num_tokens = num_tokens
        self.inner_rank = inner_rank

        #Mamba params
        self.expand = mlp_ratio
        hidden = int(self.dim * self.expand)
        self.d_state = d_state
        self.selectiveScan = Selective_Scan(d_model=hidden, d_state=self.d_state, expand=1)
        self.out_norm = nn.LayerNorm(hidden)
        self.act = nn.SiLU()
        self.out_proj = nn.Linear(hidden, dim, bias=True)

        self.in_proj = nn.Sequential(
            nn.Conv2d(self.dim, hidden, 1, 1, 0),
        )

        self.CPE = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden),
        )

        self.embeddingB = nn.Embedding(self.num_tokens, self.inner_rank)  
        self.embeddingB.weight.data.uniform_(-1 / self.num_tokens, 1 / self.num_tokens)

        self.route = nn.Sequential(
            nn.Linear(self.dim, self.dim // 3),
            nn.GELU(),
            nn.Linear(self.dim // 3, self.num_tokens),
            nn.LogSoftmax(dim=-1)
        )
        self.dstate_proj=nn.Linear(self.dim, self.d_state)
    def gather_f(self,detached_index):
        replace_values_head = [0,1,2,3,4]
        new_value_head = 0
        replace_values_right_hand = [6,8,10]
        new_value_right_hand = 1
        replace_values_left_hand = [5,7,9]
        new_value_left_hand = 2
        replace_values_right_foot = [12,14,16]
        new_value_right_foot = 3
        replace_values_left_foot= [11,13,15]
        new_value_left_foot = 4

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        mask_head = torch.isin(detached_index, torch.tensor(replace_values_head, device=device))  
        detached_index = torch.where(mask_head, new_value_head, detached_index)

        mask_right_hand = torch.isin(detached_index, torch.tensor(replace_values_right_hand, device=device))  
        detached_index = torch.where(mask_right_hand, new_value_right_hand, detached_index)

        mask_left_hand = torch.isin(detached_index, torch.tensor(replace_values_left_hand, device=device))  
        detached_index = torch.where(mask_left_hand, new_value_left_hand, detached_index)
        
        mask_right_foot = torch.isin(detached_index, torch.tensor(replace_values_right_foot, device=device))  
        detached_index = torch.where(mask_right_foot, new_value_right_foot, detached_index)

        mask_left_foot = torch.isin(detached_index, torch.tensor(replace_values_left_foot, device=device))  
        detached_index = torch.where(mask_left_foot, new_value_left_foot, detached_index)

        return detached_index
    
    def gather_t(self,detached_index):
        replace_values_head = [0,1,2,3,4]
        new_value_head = 0
        replace_values_hand = [5,6,7,8,9,10]
        new_value_hand = 1
        replace_values_foot = [11,12,13,14,15,16]
        new_value_foot = 2

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        mask_head = torch.isin(detached_index, torch.tensor(replace_values_head,device=device))  
        detached_index = torch.where(mask_head, new_value_head, detached_index)

        mask_hand = torch.isin(detached_index, torch.tensor(replace_values_hand,device=device))  
        detached_index = torch.where(mask_hand, new_value_hand, detached_index)

        mask_foot = torch.isin(detached_index, torch.tensor(replace_values_foot,device=device))  
        detached_index = torch.where(mask_foot, new_value_foot, detached_index)

        return detached_index
    
    def forward(self, x, x_size, token, gather_type):
        B, n, C = x.shape#B,HW,C
        H, W = x_size
        if gather_type!=0:
            full_embedding = self.embeddingB.weight @ token.weight  #  [17,dim]

            pred_route = self.route(x)  #B,HW,C->B, HW, num_token
            cls_policy = F.gumbel_softmax(pred_route, hard=True, dim=-1) #  [B, HW, num_token] 

            prompt = torch.matmul(cls_policy, full_embedding).view(B, n, self.d_state) #  B,HW,d_state

            detached_index = torch.argmax(cls_policy.detach(), dim=-1, keepdim=False).view(B, n)  
             
            if gather_type==3:
                detached_index=self.gather_t(detached_index)
            else:
                if gather_type==5:
                    detached_index=self.gather_f(detached_index)
            _, x_sort_indices = torch.sort(detached_index, dim=-1, stable=False)#  B,HW
            x_sort_indices_reverse = index_reverse(x_sort_indices)

            x = x.permute(0, 2, 1).reshape(B, C, H, W).contiguous()#B,C,H,W
            x = self.in_proj(x)#B,C,H,W->B,hidden,H,W
            x = x * torch.sigmoid(self.CPE(x))#B,hidden,H,W
            cc = x.shape[1]#hidden
            x = x.view(B, cc, -1).contiguous().permute(0, 2, 1)  # B,HW,hidden

            semantic_x = semantic_neighbor(x, x_sort_indices) # SGN-unfold
            y = self.selectiveScan(semantic_x, prompt,gather_type) 
            y = self.out_proj(self.out_norm(y))
            x = semantic_neighbor(y, x_sort_indices_reverse) # SGN-fold
        else:
            full_embedding = self.embeddingB.weight @ token.weight  #  [17,dim]

            pred_route = self.route(x)  #B,HW,C->B, HW, num_token
            cls_policy = F.gumbel_softmax(pred_route, hard=True, dim=-1)
            prompt = torch.matmul(cls_policy, full_embedding).view(B, n, self.d_state) 
            detached_index = torch.argmax(cls_policy.detach(), dim=-1, keepdim=False).view(B, n)  
            x = x.permute(0, 2, 1).reshape(B, C, H, W).contiguous()#B,C,H,W
            x = self.in_proj(x)#B,C,H,W->B,hidden,H,W
            x = x * torch.sigmoid(self.CPE(x))#B,hidden,H,W
            cc = x.shape[1]#hidden
            x = x.view(B, cc, -1).contiguous().permute(0, 2, 1)  # B,HW,hidden

            #semantic_x = semantic_neighbor(x, x_sort_indices) # SGN-unfold
            y = self.selectiveScan(x, prompt,gather_type) 
            x = self.out_proj(self.out_norm(y))
            #x = semantic_neighbor(y, x_sort_indices_reverse) # SGN-fold
        return x


class VSSBlock(nn.Module):
    def __init__(self,
                 dim,
                 d_state=16,
                 inner_rank=64,
                 num_tokens=17,
                 convffn_kernel_size=5,
                 mlp_ratio=2.,
                 norm_layer=nn.LayerNorm,
                 ):
        super().__init__()

        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.convffn_kernel_size = convffn_kernel_size
        self.num_tokens = num_tokens
        self.inner_rank = inner_rank

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)


        self.assm = ASSM(
            self.dim,
            d_state,
            num_tokens=num_tokens,
            inner_rank=inner_rank,
            mlp_ratio=mlp_ratio
        )

        mlp_hidden_dim = int(dim * self.mlp_ratio)
        self.convffn = ConvFFN(in_features=dim, hidden_features=mlp_hidden_dim, kernel_size=convffn_kernel_size,)
        self.embeddingA = nn.Embedding(self.inner_rank,d_state)
        self.embeddingA.weight.data.uniform_(-1 / self.inner_rank, 1 / self.inner_rank)
        #self.keypoint_proj = nn.Linear(d_model,d_model*2)

    def forward(self, x, x_size,gather_type):
        x_aca = self.assm(self.norm1(x), x_size, self.embeddingA,gather_type) + x
        x = x_aca + self.convffn(self.norm2(x_aca), x_size)

        return x
    
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

     
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
      

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

   
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

    

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
 

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ASMPose(nn.Module):

    def __init__(self, block, layers, cfg, **kwargs):
        self.inplanes = 64
        extra = cfg.MODEL.EXTRA
        self.deconv_with_bias = extra.DECONV_WITH_BIAS
        
        super(ASMPose, self).__init__()
        
        d_model = cfg.MODEL.DIM_MODEL
        dim_feedforward = cfg.MODEL.DIM_FEEDFORWARD
        encoder_layers_num = cfg.MODEL.ENCODER_LAYERS
        n_head = cfg.MODEL.N_HEAD
        pos_embedding_type = cfg.MODEL.POS_EMBEDDING
        w, h = cfg.MODEL.IMAGE_SIZE
        
        d_1=d_model//8
        d_2=d_model//4
        d_3=d_model//2
        self.d_model=d_model
        self.d_1=d_1
        self.d_2=d_2
        self.d_3=d_3
        self.conv1 = nn.Conv2d(3, d_1, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(d_1, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(in_channels= d_1,out_channels= d_1,kernel_size=3,stride=1,padding=1,groups= d_1)
        self.bn2 = nn.BatchNorm2d(d_1, momentum=BN_MOMENTUM)
        
        self.conv3 = nn.Conv2d(in_channels= d_1,out_channels= d_1,kernel_size=3,stride=1,padding=1,groups= d_1)
        self.bn3 = nn.BatchNorm2d( d_1, momentum=BN_MOMENTUM)
        
        self.conv4 = nn.Conv2d(in_channels= d_1,out_channels= d_2,kernel_size=3,stride=2,padding=1,groups= d_1)
        
        self.conv5 = nn.Conv2d(in_channels=d_2,out_channels=d_3,kernel_size=3,stride=2,padding=1,groups=d_2)
        
        self.conv6 = nn.Conv2d(in_channels=d_3,out_channels=d_model,kernel_size=3,stride=2,padding=1,groups=d_3)
        

        self.ssm1_1=VSSBlock_gather_4(hidden_dim=d_2,keypoint_num=17,inner_rank=64,d_token=d_2)
        self.ssm1_2=VSSBlock_gather_4(hidden_dim=d_2,keypoint_num=17,inner_rank=64,d_token=d_2)
        
        self.ssm2_1=VSSBlock_gather_4(hidden_dim=d_3,keypoint_num=17,inner_rank=64,d_token=d_3)
        self.ssm2_2=VSSBlock_gather_4(hidden_dim=d_3,keypoint_num=17,inner_rank=64,d_token=d_3)
        self.ssm2_3=VSSBlock_gather_4(hidden_dim=d_3,keypoint_num=17,inner_rank=64,d_token=d_3)
        self.ssm2_4=VSSBlock_gather_4(hidden_dim=d_3,keypoint_num=17,inner_rank=64,d_token=d_3)
        
        self.ssm3_1=VSSBlock_gather_4(hidden_dim=d_model,keypoint_num=17,inner_rank=64,d_token=d_model)
        self.ssm3_2=VSSBlock_gather_4(hidden_dim=d_model,keypoint_num=17,inner_rank=64,d_token=d_model)
        self.ssm3_3=VSSBlock_gather_4(hidden_dim=d_model,keypoint_num=17,inner_rank=64,d_token=d_model)
        self.ssm3_4=VSSBlock_gather_4(hidden_dim=d_model,keypoint_num=17,inner_rank=64,d_token=d_model)
        self.ssm3_5=VSSBlock_gather_4(hidden_dim=d_model,keypoint_num=17,inner_rank=64,d_token=d_model)
        self.ssm3_6=VSSBlock_gather_4(hidden_dim=d_model,keypoint_num=17,inner_rank=64,d_token=d_model)
        
        self.downsample_1 = nn.Sequential(
                nn.Conv2d(in_channels= d_2,out_channels=d_model,kernel_size=3,stride=2,padding=1,groups= d_2),
                nn.BatchNorm2d(d_model, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
   
            )
        #128,32,24->256,16,12
        self.downsample_2 = nn.Sequential(
                nn.Conv2d(in_channels=d_3,out_channels=d_model,kernel_size=3,stride=2,padding=1,groups=d_3),
                nn.BatchNorm2d(d_model, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
          
       
            )
        
        
        '''
        self._make_position_embedding1(w, h, d_3, pos_embedding_type)
        self._make_position_embedding2(w//2, h//2, d_model, pos_embedding_type)
        '''

        
        # used for deconv layers
        self.inplanes = d_model
        self.deconv_layers_1 = self._make_deconv_layer(
            1,   # 1
            extra.NUM_DECONV_FILTERS,  # [d_model]
            extra.NUM_DECONV_KERNELS,  # [4]
        )
        
        self.deconv_layers_2 = self._make_deconv_layer(
            1,   # 1
            extra.NUM_DECONV_FILTERS,  # [d_model]
            extra.NUM_DECONV_KERNELS,  # [4]
        )
    
        self.final_layer = nn.Conv2d(
            in_channels=d_model,
            out_channels=cfg.MODEL.NUM_JOINTS,
            kernel_size=extra.FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0
        )
    def _make_position_embedding1(self, w, h, d_model, pe_type='sine'):
        assert pe_type in ['none', 'learnable', 'sine']
        if pe_type == 'none':
            self.pos_embedding1 = None
            logger.info("==> Without any PositionEmbedding~")
        else:
            with torch.no_grad():
          
             
                self.pe_h = h // 16
                self.pe_w = w // 16
                length = self.pe_h * self.pe_w
            if pe_type == 'learnable':
                self.pos_embedding1 = nn.Parameter(
                    torch.randn(length, 1, d_model))
                logger.info("==> Add Learnable PositionEmbedding~")
            else:
                self.pos_embedding1 = nn.Parameter(
                    self._make_sine_position_embedding1(d_model),
                    requires_grad=False)
                logger.info("==> Add Sine PositionEmbedding~")

    def _make_position_embedding2(self, w, h, d_model, pe_type='sine'):
        assert pe_type in ['none', 'learnable', 'sine']
        if pe_type == 'none':
            self.pos_embedding2 = None
            logger.info("==> Without any PositionEmbedding~")
        else:
            with torch.no_grad():

 
                self.pe_h = h // 32
                self.pe_w = w // 32
                length = self.pe_h * self.pe_w
            if pe_type == 'learnable':
                self.pos_embedding2 = nn.Parameter(
                    torch.randn(length, 1, d_model))
                logger.info("==> Add Learnable PositionEmbedding~")
            else:
                self.pos_embedding2 = nn.Parameter(
                    self._make_sine_position_embedding2(d_model),
                    requires_grad=False)
                logger.info("==> Add Sine PositionEmbedding~")

    def _make_sine_position_embedding1(self, d_model, temperature=10000,
                                      scale=2*math.pi):
        # logger.info(">> NOTE: this is for testing on unseen input resolutions")
        # # NOTE generalization test with interploation
        # self.pe_h, self.pe_w = 256 // 8 , 192 // 8 #self.pe_h, self.pe_w
        h, w = self.pe_h, self.pe_w
        area = torch.ones(1, h, w)  # [b, h, w]
        y_embed = area.cumsum(1, dtype=torch.float32)
        x_embed = area.cumsum(2, dtype=torch.float32)

        one_direction_feats = d_model // 2

        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

        dim_t = torch.arange(one_direction_feats, dtype=torch.float32)
        dim_t = temperature ** (2 * (dim_t // 2) / one_direction_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2).contiguous()
        pos = pos.flatten(2).permute(2, 0, 1).contiguous()
        return pos  # [h*w, 1, d_model]

    def _make_sine_position_embedding2(self, d_model, temperature=10000,
                                      scale=2*math.pi):
        # logger.info(">> NOTE: this is for testing on unseen input resolutions")
        # # NOTE generalization test with interploation
        # self.pe_h, self.pe_w = 256 // 8 , 192 // 8 #self.pe_h, self.pe_w
        h, w = self.pe_h, self.pe_w
        area = torch.ones(1, h, w)  # [b, h, w]
        y_embed = area.cumsum(1, dtype=torch.float32)
        x_embed = area.cumsum(2, dtype=torch.float32)

        one_direction_feats = d_model // 2

        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

        dim_t = torch.arange(one_direction_feats, dtype=torch.float32)
        dim_t = temperature ** (2 * (dim_t // 2) / one_direction_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2).contiguous()
        pos = pos.flatten(2).permute(2, 0, 1).contiguous()
        return pos  # [h*w, 1, d_model]

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):

       
        x = self.conv1(x) 
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x) 
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.conv3(x) 
        x = self.bn3(x)
        x = self.relu(x)
        
        
        x = self.conv4(x) #B,64,H/4,W/4

    
        x = self.ssm1_1(x,0)
        x = self.ssm1_2(x,0)
        
        
        
        short_1 = self.downsample_1(x)  #64,64,48->256,32,24
        x = self.conv5(x) #B,64,H/4,W/4->B,128,H/8,W/8


        x = self.ssm2_1(x,3)
        x = self.ssm2_2(x,3)
        x = self.ssm2_3(x,3)
        x = self.ssm2_4(x,3)
       

        short_2 = self.downsample_2(x) #128,32,24->256,16,12
        x = self.conv6(x) #B,128,H/8,W/8->B,256,H/16,W/16

        
        x = self.ssm3_1(x,5)
        x = self.ssm3_2(x,5)
        x = self.ssm3_3(x,5)
        x = self.ssm3_4(x,5)
        x = self.ssm3_5(x,5)
        x = self.ssm3_6(x,5)


        x = short_2 + x
        x = self.deconv_layers_1(x)
        
        #256,32,24
        x = short_1 + x
        x = self.deconv_layers_2(x)

   
        x = self.final_layer(x) 

   

        return x

    def init_weights(self, pretrained=''):
        if os.path.isfile(pretrained):
            logger.info('=> init final conv weights from normal distribution')
            for name, m in self.final_layer.named_modules():
                if isinstance(m, nn.Conv2d):
                    logger.info(
                        '=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    nn.init.constant_(m.bias, 0)

            pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            existing_state_dict = {}
            for name, m in pretrained_state_dict.items():
                if name in self.state_dict():
                    if 'pos_embedding' not in name:  #del not equal  para
                        if 'final_layer.bias' not in name:
                            if 'final_layer.weigh' not in name:
                                existing_state_dict[name] = m
                    print(":: {} is loaded from {}".format(name, pretrained))
            self.load_state_dict(existing_state_dict, strict=False)
        else:
            logger.info(
                '=> NOTE :: ImageNet Pretrained Weights {} are not loaded ! Please Download it'.format(pretrained))
            logger.info('=> init weights from normal distribution')
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, std=0.001)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.ConvTranspose2d):
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)


resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
               34: (BasicBlock, [3, 4, 6, 3]),
               50: (Bottleneck, [3, 4, 6, 3]),
               101: (Bottleneck, [3, 4, 23, 3]),
               152: (Bottleneck, [3, 8, 36, 3])}


def get_pose_net(cfg, is_train, **kwargs):
    num_layers = cfg.MODEL.EXTRA.NUM_LAYERS
    block_class, layers = resnet_spec[num_layers]
    model = ASMPose(block_class, layers, cfg, **kwargs)

    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED)

    return model
