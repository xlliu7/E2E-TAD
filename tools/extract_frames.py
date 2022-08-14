from re import sub
import numpy as np
import os.path as osp
import concurrent.futures
import os
import argparse
import json


def extract_frames(video_path, dst_dir, fps):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    video_fname = osp.basename(video_path)
    
    if not osp.exists(video_path):
        subdir = 'test_set/TH14_test_set_mp4' if 'test' in video_fname else 'Validation_set/videos'
        url = f'https://crcv.ucf.edu/THUMOS14/{subdir}/{video_fname}'
        os.system('wget {} -O {} --no-check-certificate'.format(url, video_path))
    cmd = 'ffmpeg -i "{}"  -filter:v "fps=fps={}" "{}/img_%07d.jpg"'.format(video_path, fps, dst_dir)
    print(cmd)
    ret_code = os.system(cmd)
    if ret_code == 0:
        os.system('touch logs/frame_extracted_{}fps/{}'.format(fps, osp.splitext(osp.basename(video_path))[0]))
    return ret_code == 0


def parse_args():
    parser = argparse.ArgumentParser('Extract frames')
    parser.add_argument('--video_dir', help='path to the parent dir of video directory')
    parser.add_argument('--frame_dir', help='path to save extracted video frames')
    parser.add_argument('--fps', type=int, default=10)
    parser.add_argument('-s', '--start', type=int)
    parser.add_argument('-e', '--end', type=int)

    args = parser.parse_args()
    return args



def mkdir_if_missing(dirname):
    if not osp.exists(dirname):
        os.makedirs(dirname)

def main(subset):
    args = parse_args()

    log_dir = 'logs/frame_extracted_{}fps'.format(args.fps)
    mkdir_if_missing(log_dir)
    mkdir_if_missing(args.video_dir)

    database = json.load(open('data/thumos14/th14_annotations_with_fps_duration.json'))['database']
    vid_names = list(sorted([x for x in database if database[x]['subset'] == subset]))

    start_ind = 0 if args.start is None else args.start
    end_ind = len(vid_names) if args.end is None else min(args.end, len(vid_names))

    vid_names = vid_names[args.start:args.end]

    finished = os.listdir('logs/frame_extracted_{}fps'.format(args.fps))
    videos_todo = list(sorted(set(vid_names).difference(finished)))
    with concurrent.futures.ProcessPoolExecutor(4) as f:
        futures = [f.submit(extract_frames, osp.join(args.video_dir, x + '.mp4'),
                            osp.join(args.frame_dir, x), args.fps) for x in videos_todo]

    for f in futures:
        f.result()


if __name__ == '__main__':
    main('val')
    main('test')

# thumos14
# python tools/extract_frames.py --video_dir data/thumos14/videos --frame_dir data/thumos14/img10fps --fps  10 -e 4