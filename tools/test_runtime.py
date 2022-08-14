
import argparse
import time
import pdb
import logging

import numpy as np
import torch
from torch import nn

from models import build_model
from util.misc import NestedTensor
from opts import get_args_parser, update_cfg_with_args, update_cfg_from_file, cfg
from .flop_count import flop_count
from datasets.data_utils import get_dataset_info, get_dataset_dict

import warnings
warnings.filterwarnings("ignore")


logging.getLogger().setLevel(logging.INFO)


def get_slice_and_video_num(args):
    subset_mapping, _, ann_file, meta_file = get_dataset_info(args.dataset_name, args.feature)  #get_dataset_info('hacs', 'i3d')
    dataset_dict = get_dataset_dict(meta_file, ann_file, subset_mapping['val'], online_slice=args.online_slice, slice_len=args.slice_len, ignore_empty=False, slice_overlap=args.test_slice_overlap)
    video_num = len(set(v['src_vid_name'] for v in dataset_dict.values()))
    return len(dataset_dict), video_num


@torch.no_grad()
def run():
    import os
    parser = argparse.ArgumentParser('TadTR timing toolkit', parents=[get_args_parser()])
    parser.add_argument('--input_length', type=int, default=100)
    parser.add_argument('--no_head', action='store_true')
    parser.add_argument('--gpu', type=int)
    args = parser.parse_args()

    if args.cfg is not None:
        update_cfg_from_file(cfg, args.cfg)

    update_cfg_with_args(cfg, args.opt)

    model, criterion, postprocessors = build_model(cfg)
    if args.no_head:
        model = model.backbone.backbone
    # print(model)

    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        device_id = args.gpu[0] if args.gpu is not None else 0
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    model.cuda()
    model.eval()
    # if not args.quiet:
    #     print(model)
    
    # x = torch.rand((1,3,128, 112, 112)).cuda()
    if cfg.online_slice:
        input_length = cfg.slice_len
    else:
        if args.input_length is not None:
            input_length = args.input_length
        else:
            raise ValueError('input_length is not specified')
            # input_length = 100 if args.input_type == 'feature' else 768
    if cfg.input_type == 'feature':
        input = torch.rand((1, cfg.feature_dim, input_length))
        mask = torch.zeros((1, input_length), dtype=torch.bool, device=input.device)

    else:
        # input_length = 384
        stride = cfg.img_stride
        input =  torch.rand((1, 3, input_length//stride, cfg.img_crop_size, cfg.img_crop_size))
        mask = torch.zeros((1, input_length//stride), dtype=torch.bool, device=input.device)
    
    input = input.cuda()
    mask = mask.cuda()
    input_nt = NestedTensor(input, mask)
    if args.no_head:
        input_nt = input

    do_runtime_test, do_flops_test = True, True
    slice_num, video_num = get_slice_and_video_num(cfg)

    def warmup(model, inputs, N=10):
        for i in range(N):
            out = model(inputs)
        torch.cuda.synchronize()

    if do_runtime_test:
        step = 100
        input_cuda = input_nt.to(torch.device('cuda'))

        warmup(model, input_cuda, 10)
        
        t0 = time.time()
        for _ in range(step):
            y = model(input_cuda)
        torch.cuda.synchronize()
        t2 = time.time()

        delta = (t2 - t0) / step
        # delta = np.mean(time_list[5:])
        fps = input_length / delta  
        print('{:.6f}s/slice {:.6f}s/video, {:.0f}fps, averaged over {} runs (after 10 warmup runs)'.format(delta, delta * slice_num/video_num, fps, step))


    if do_flops_test:
        res = flop_count(model, (input,))              
        gflops = sum(res.values())
        print('flops={:.6f}G/slice {:.6f}G/video'.format(gflops, gflops*slice_num/video_num))

if __name__ == '__main__':
    run()

# 