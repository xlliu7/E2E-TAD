"""
Video encoder modules.
"""
import ipdb as pdb
from collections import OrderedDict
from typing import Dict, List
import logging

import torch
import torch.nn.functional as F
from torch.nn.modules.normalization import GroupNorm
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from einops import rearrange

from models.video_encoder_archs.tsm import TSM
from models.video_encoder_archs.slowfast import ResNet3dSlowFast
from util.misc import NestedTensor, is_main_process
from opts import cfg


def unfold(ip, kernel_size, stride):
    '''Expect NCTHW shaped tensor, extract sliding block for snippet-wise feature extraction'''
    # ip_ncts = rearrange(ip_ncthw, "n c t h w -> n c t (h w)")
    # ip_ncts = F.unfold(ip_ncts, (kernel_size, 1), stride=(stride, 1), padding=((kernel_size-stride)//2, 1))
    N, C, T, H, W = ip.shape
    pad_size = (( kernel_size - stride ) // 2, (kernel_size-stride+1) // 2)
    ip_pad = F.pad(ip, (0, 0, 0, 0, *pad_size), mode='constant', value=0)
    num_windows = T // stride
    start = torch.arange(num_windows).reshape([num_windows, 1]) * stride
    indices = (start + torch.arange(kernel_size)).view(-1)  # (num_windows, kernel_size)
    out = torch.index_select(ip_pad, dim=2, index=indices.to(ip.device))
    # pdb.set_trace()
    out= out.reshape(N, C, num_windows, kernel_size, H, W)
    out = rearrange(out, 'n c nw ks h w -> (n nw) c ks h w')
    return out


class VideoEncoder(nn.Module):
    def __init__(self, arch='slowfast', fix_encoder=False):
        super().__init__()
        self.arch = arch
        self.use_upsample = cfg.temporal_upsample
        
        if arch == 'slowfast':
            self.backbone = ResNet3dSlowFast(None, depth=cfg.slowfast_depth,freeze_bn=cfg.freeze_bn, freeze_bn_affine=cfg.freeze_affine, slow_upsample=cfg.slow_upsample)
            self.num_channels = 2304

        elif arch in ['tsm', 'tsn']:    
            self.backbone = TSM(arch=cfg.tsm_base_model, is_shift=arch=='tsm')
            self.num_channels = self.backbone.out_channels

        else:
            raise ValueError('Not supported arch: {}'.format(arch))
        
        self.fix_encoder = fix_encoder
        
        if fix_encoder:
            self._fix_encoder()
        

    def forward(self, tensor_list):
        '''tensor_list: tensors+mask'''
        if not isinstance(tensor_list, NestedTensor):
            b, t = tensor_list.shape[0], tensor_list.shape[2]
            mask = torch.zeros((b, t), dtype=torch.bool, device=tensor_list.device)
            tensor_list = NestedTensor(tensor_list, mask)
        tensors = tensor_list.tensors
        batch_size = tensors.shape[0]
        mask = tensor_list.mask
        shape = tensors.shape

        # it takes as input image sequence or feature vector sequence
        if len(shape) == 5:   # (n,c,t,h,w)
            pooler = F.adaptive_max_pool3d if cfg.spatial_pool == 'max' else F.adaptive_avg_pool3d
           
            ip = tensor_list.tensors
            if cfg.snippet_wise_feature:
                ip = unfold(tensor_list.tensors, cfg.snippet_length, cfg.snippet_stride)
                video_ft = self.backbone(ip).mean(2)       # (n*n_window, c, t, h, w)
                T = video_ft.shape[0] // batch_size
                video_ft_fold = video_ft.reshape(batch_size, T, *(video_ft.shape[1:]))  # (n, n_window, c, h, w)
                video_ft = video_ft_fold.transpose(1, 2)
            else:
                # fully convolutional feature extraction
                video_ft = self.backbone(tensor_list.tensors)  # [n,c,t, h, w]

            if isinstance(video_ft, (list, tuple)) and len(video_ft) == 1:
                video_ft = video_ft[0]

            if not isinstance(video_ft, (list, tuple)):
                if video_ft.ndim == 5:
                    video_ft = pooler(video_ft, [None, 1, 1])[..., 0, 0]  # [n, c, t]
                if self.use_upsample:
                    video_ft = F.interpolate(video_ft, scale_factor=cfg.temporal_upscale, mode='linear')
                mask = F.interpolate(mask[None].float(), size=video_ft.shape[2], mode='nearest').to(torch.bool)[0]  # [n, t]
                out = NestedTensor(video_ft, mask)
            else:
                # multilevel feature from backbone
                raise NotImplementedError

        elif len(shape) == 3: # (n,c,t)
            video_ft = tensors
            out = NestedTensor(video_ft, mask)
        
        return out

    def _fix_encoder(self):
        logging.info('freezing the backbone')
        self.backbone.requires_grad_(False)
        

class EmptyEncoder(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.num_channels = feature_dim

    def forward(self, x):
        return x


def build_video_encoder(args):
    if args.input_type == 'feature':
        model = EmptyEncoder(args.feature_dim)
    else:
        model = VideoEncoder(args.encoder, args.fix_encoder)
    return model
