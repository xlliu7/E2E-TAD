import os

import numpy as np
import cv2


def load_video_frames(frame_dir, start, seq_len, stride=1, fn_tmpl='img_%07d.jpg'):
    '''
    Load a sequence of video frames into memory. 

    Params:
        frame_dir: the directory to the decoded video frames
        start: load image starting with this index. 1-indexed
        seq_len: length of the loaded sub-sequence.
        stride: load one frame every `stride` frame from the sequence.
    Returns:
        Nd-array with shape (T, H, W, C) in float32 precision. T = num // stride
    '''
    frames = []
    if seq_len > 0:
        # load a fixed-length frame sequence
        # for i in range(start + stride // 2, start + seq_len, stride):
        #     img = cv2.imread(os.path.join(frame_dir, fn_tmpl % i))
        #     if  img is None:
        #         # print('failed to load {}'.format(os.path.join(frame_dir, fn_tmpl % i)))
        #         raise IOError(os.path.join(frame_dir, fn_tmpl % i))
        #     # img = img[:, :, [2, 1, 0]]  # BGR => RGB, moved to video_transforms.Normalize
        #     # img = (img/255.)*2 - 1
        #     frames.append(img)
        frames = [cv2.imread(os.path.join(frame_dir, fn_tmpl % i))
            for i in range(start + stride // 2, start + seq_len, stride)]
    else:
        # load all frames
        num_imgs = len(os.listdir(frame_dir))
        frames = [cv2.imread(os.path.join(frame_dir, fn_tmpl % (i+1))) for i in range(num_imgs)]
    return np.asarray(frames, dtype=np.float32)  # NHWC


def make_img_transform(is_training, resize=110, crop=96, mean=127.5, std=127.5, keep_asr=True):
    from .videotransforms import GroupResizeShorterSide, GroupCenterCrop, GroupRandomCrop, GroupRandomHorizontalFlip, GroupPhotoMetricDistortion, GroupRotate, GroupResize, GroupNormalize
    from torchvision.transforms import Compose

    if isinstance(resize, (list, tuple)):
        resize_trans = GroupResize(resize)
    else:
        if keep_asr:
            assert isinstance(resize, int), 'if keep asr, resize must be a single integer'
            resize_trans = GroupResizeShorterSide(resize)
        else:
            resize_trans = GroupResize((resize, resize))

    transforms = [
        resize_trans,
        GroupRandomCrop(crop) if is_training else GroupCenterCrop(crop),
    ]
    if is_training:
            transforms += [
                GroupPhotoMetricDistortion(brightness_delta=32,
                    contrast_range=(0.5, 1.5),
                    saturation_range=(0.5, 1.5),
                    hue_delta=18,
                    p=0.5),
                GroupRotate(limit=(-45, 45),
                    border_mode='reflect101',
                    p=0.5),
                GroupRandomHorizontalFlip(0.5),
            ]
        
    transforms.append(GroupNormalize(mean, std, to_rgb=True))
    return Compose(transforms)