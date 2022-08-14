# Mostly copied from the official repo of TSM paper https://github.com/mit-han-lab/temporal-shift-module/blob/master/ops/models.py

import sys
import os
import logging

import torch
from torch import nn
import torch.nn.functional as F

import torchvision


dirname = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(dirname, '..'))
from opts import cfg


# this variable is helpful for recover the video sequence from batched images
BATCH_SIZE = None   
VERBOSE = False


class TemporalShift(nn.Module):
    def __init__(self, net, n_div=8, inplace=False):
        super(TemporalShift, self).__init__()
        self.net = net
        self.fold_div = n_div
        self.inplace = inplace
        if inplace:
            if VERBOSE:  print('=> Using in-place shift...')
        if VERBOSE:  print('=> Using fold div: {}'.format(self.fold_div))

    def forward(self, x):
        x = self.shift(x, fold_div=self.fold_div, inplace=self.inplace)
        return self.net(x)

    @staticmethod
    def shift(x, fold_div=3, inplace=False):
        nt, c, h, w = x.size()
        # n_batch = nt // n_segment
        n_batch = BATCH_SIZE
        n_segment = nt // n_batch
        x = x.view(n_batch, n_segment, c, h, w)

        fold = c // fold_div
        if inplace:
            # Due to some out of order error when performing parallel computing. 
            # May need to write a CUDA kernel.
            raise NotImplementedError  
            # out = InplaceShift.apply(x, fold)
        else:
            out = torch.zeros_like(x)
            out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
            out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
            out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift

        return out.view(nt, c, h, w)


class InplaceShift(torch.autograd.Function):
    # Special thanks to @raoyongming for the help to this function
    @staticmethod
    def forward(ctx, input, fold):
        # not support higher order gradient
        # input = input.detach_()
        ctx.fold_ = fold
        n, t, c, h, w = input.size()
        buffer = input.data.new(n, t, fold, h, w).zero_()
        buffer[:, :-1] = input.data[:, 1:, :fold]
        input.data[:, :, :fold] = buffer
        buffer.zero_()
        buffer[:, 1:] = input.data[:, :-1, fold: 2 * fold]
        input.data[:, :, fold: 2 * fold] = buffer
        return input

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output = grad_output.detach_()
        fold = ctx.fold_
        n, t, c, h, w = grad_output.size()
        buffer = grad_output.data.new(n, t, fold, h, w).zero_()
        buffer[:, 1:] = grad_output.data[:, :-1, :fold]
        grad_output.data[:, :, :fold] = buffer
        buffer.zero_()
        buffer[:, :-1] = grad_output.data[:, 1:, fold: 2 * fold]
        grad_output.data[:, :, fold: 2 * fold] = buffer
        return grad_output, None


class TemporalPool(nn.Module):
    def __init__(self, net):
        super(TemporalPool, self).__init__()
        self.net = net

    def forward(self, x):
        x = self.temporal_pool(x)
        return self.net(x)

    @staticmethod
    def temporal_pool(x):
        nt, c, h, w = x.size()
        n_batch = BATCH_SIZE
        n_segment = nt // n_batch
        x = x.view(n_batch, n_segment, c, h, w).transpose(1, 2)  # n, c, t, h, w
        x = F.max_pool3d(x, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0))
        x = x.transpose(1, 2).contiguous().view(nt // 2, c, h, w)
        return x

    

def make_temporal_shift(net, n_div=8, place='blockres'):
 
    # import torchvision
    # if isinstance(net, torchvision.models.ResNet):
    if place == 'block':
        def make_block_temporal(stage):
            blocks = list(stage.children())
            if VERBOSE:  print('=> Processing stage with {} blocks'.format(len(blocks)))
            for i, b in enumerate(blocks):
                blocks[i] = TemporalShift(b, n_div=n_div)
            return nn.Sequential(*(blocks))

        net.layer1 = make_block_temporal(net.layer1)
        net.layer2 = make_block_temporal(net.layer2)
        net.layer3 = make_block_temporal(net.layer3)
        net.layer4 = make_block_temporal(net.layer4)

    # default
    elif 'blockres' in place:
        n_round = 1
        if len(list(net.layer3.children())) >= 23:
            n_round = 2
            if VERBOSE:  print('=> Using n_round {} to insert temporal shift'.format(n_round))

        def make_block_temporal(stage):
            blocks = list(stage.children())
            if VERBOSE:  print('=> Processing stage with {} blocks residual'.format(len(blocks)))
            for i, b in enumerate(blocks):
                if i % n_round == 0:
                    blocks[i].conv1 = TemporalShift(b.conv1, n_div=n_div)
            return nn.Sequential(*blocks)

        net.layer1 = make_block_temporal(net.layer1)
        net.layer2 = make_block_temporal(net.layer2)
        net.layer3 = make_block_temporal(net.layer3)
        net.layer4 = make_block_temporal(net.layer4)
   

def make_temporal_pool(net, n_segment):
    import torchvision
    if isinstance(net, torchvision.models.ResNet):
        if VERBOSE:  print('=> Injecting nonlocal pooling')
        net.layer2 = TemporalPool(net.layer2, n_segment)
    else:
        raise NotImplementedError


class TSM(nn.Module):
    def __init__(self, arch='resnet50', is_shift=True, shift_div=8, shift_place='blockres', temporal_pool=False, freeze_bn=True):
        '''
        arch: the architecture of the backbone, e.g. resnet50
        is_shift: whether to enable the shift module. If set false, downgrade to TSN (not supported).
        freeze_bn: whether to freeze the batch norm layers
        '''
        super().__init__()
        logging.info('ResNet arch={}'.format(arch))
        self.arch = arch
        self.is_shift = is_shift
        self.shift_div = shift_div
        self.shift_place = shift_place
        self.temporal_pool = temporal_pool
        self._freeze_bn = freeze_bn
        self._freeze_bn_affine = cfg.freeze_affine
        self.frozen_stages = cfg.frozen_stages

        self.out_channels = 2048 if int(arch[len('resnet'):]) >= 50 else 512

        assert is_shift, "is_shift should be set True"

        if 'resnet' in arch:
            channels = [self.out_channels // (2**i) for i in range(3)]
            self.per_stage_out_channels = channels

            base_net = getattr(torchvision.models, arch)(pretrained=False)
            for name, module in base_net.named_children():
                if name == 'fc' or name == 'avgpool':
                    continue
                self.add_module(name, module)

            if is_shift:
                make_temporal_shift(self, 
                                    n_div=self.shift_div, place=self.shift_place)
        else:
            raise ValueError("Unsupported arch {}".format(arch))
        

    def extract_features(self, x):
        '''Extract 5D feature maps'''
        N, C, T, H, W = x.shape
         
        global BATCH_SIZE
        BATCH_SIZE = N
        x = x.transpose(1, 2).flatten(0, 1).contiguous()  # (n*t, c, h, w)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)     # (n*t,c,h,w)
        NT, C, H, W = x.shape
        x = x.reshape([N, NT//N, C, H, W]).transpose(1, 2).contiguous()  # (n,c,t, h, w)
        return x

    def forward(self, x):
        y = self.extract_features(x)
        return y

    def load_pretrained_weight(self, ckpt_path=None):
        if ckpt_path is None:
            ckpt_path = 'https://hanlab.mit.edu/projects/tsm/models/TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e100_dense.pth'
        
        logging.info('loading pretrained model {}'.format(ckpt_path))

        if ckpt_path.startswith('http'):
            checkpoint = torch.hub.load_state_dict_from_url(
                ckpt_path, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(ckpt_path)
        checkpoint = checkpoint['state_dict']

        base_dict = {'.'.join(k.split('.')[2:]): v for k, v in list(checkpoint.items())}
        
        self.load_state_dict(base_dict, strict=False)

    def train(self, mode=True):
        super().train(mode)
        if self.frozen_stages >= 0:
            logging.info('freeze 0 to {} stage (0-index)'.format(self.frozen_stages))
        self._freeze_stages()
        
        if self._freeze_bn and mode:
            for name, m in self.named_modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    if self._freeze_bn_affine:
                        m.weight.requires_grad_(False)
                        m.bias.requires_grad_(False)
                

    def _freeze_stages(self):
        """Prevent all the parameters from being optimized before
        ``self.frozen_stages``."""
        if self.frozen_stages >= 0:
            self.conv1.requires_grad_(False)
            self.conv1.eval()
            for param in self.conv1.parameters():
                param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.requires_grad_(False)
            m.eval()
            for param in m.parameters():
                param.requires_grad = False


def test_shift():
    # test inplace shift v.s. vanilla shift
    tsm1 = TemporalShift(nn.Sequential(), n_segment=8, n_div=8, inplace=False)
    tsm2 = TemporalShift(nn.Sequential(), n_segment=8, n_div=8, inplace=True)

    print('=> Testing CPU...')
    # test forward
    with torch.no_grad():
        for i in range(10):
            x = torch.rand(2 * 8, 3, 224, 224)
            y1 = tsm1(x)
            y2 = tsm2(x)
            assert torch.norm(y1 - y2).item() < 1e-5

    # test backward
    with torch.enable_grad():
        for i in range(10):
            x1 = torch.rand(2 * 8, 3, 224, 224)
            x1.requires_grad_()
            x2 = x1.clone()
            y1 = tsm1(x1)
            y2 = tsm2(x2)
            grad1 = torch.autograd.grad((y1 ** 2).mean(), [x1])[0]
            grad2 = torch.autograd.grad((y2 ** 2).mean(), [x2])[0]
            assert torch.norm(grad1 - grad2).item() < 1e-5

    print('=> Testing GPU...')
    tsm1.cuda()
    tsm2.cuda()
    # test forward
    with torch.no_grad():
        for i in range(10):
            x = torch.rand(2 * 8, 3, 224, 224).cuda()
            y1 = tsm1(x)
            y2 = tsm2(x)
            assert torch.norm(y1 - y2).item() < 1e-5

    # test backward
    with torch.enable_grad():
        for i in range(10):
            x1 = torch.rand(2 * 8, 3, 224, 224).cuda()
            x1.requires_grad_()
            x2 = x1.clone()
            y1 = tsm1(x1)
            y2 = tsm2(x2)
            grad1 = torch.autograd.grad((y1 ** 2).mean(), [x1])[0]
            grad2 = torch.autograd.grad((y2 ** 2).mean(), [x2])[0]
            assert torch.norm(grad1 - grad2).item() < 1e-5
    print('Test passed.')



def measure_time(model, x, repeat=20, warmup=10):
    for _ in range(warmup):
        y = model(x)
    
    s = time.time()
    for _ in range(repeat):
        y = model(x)
    torch.cuda.synchronize()
    diff = time.time() - s
    return diff/repeat


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    import time
    import numpy as np
    s = time.time()
    # from flop_count import flop_count
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    is_shift = True
    
    model = TSM(arch='resnet18', is_shift=is_shift).cuda()
    
    model.eval()
    model.requires_grad_(False)

    stride = 8
    time_list = []
    x = torch.rand([1, 3, 32, 96, 96]).cuda()

    # flops = flop_count(model, (x,))
    # flops = sum(flops.values())
    flops = 0

    time_cost = measure_time(model, x, repeat=20, warmup=10)
    memory = torch.cuda.max_memory_allocated() / 1024 / 1024
    
    print('Time {:.2f}ms {:.0f}M memory {:.1f}GFLOPS'.format(time_cost*1000, memory, flops))
    