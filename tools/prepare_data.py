import os
import json
import os.path as osp


def gen_thumos14_frames_info(frame_dir, fps):
    files = os.listdir(frame_dir)
    result_dict = {}
    anno_dict = json.load(open(osp.expanduser('data/thumos14/th14_annotations_with_fps_duration.json')))['database']

    for fname in files:
        vid = fname
        num_frames = len(os.listdir(osp.join(frame_dir, vid)))
        feature_second = num_frames / fps
        video_second = anno_dict[vid]['duration']
        diff = abs(feature_second - video_second)
        if diff > 3:
            print(fname, feature_second, video_second)
        feature_fps = fps     # anno_dict[vid]['fps'] / 8
        result_dict[vid] = {'feature_length': num_frames, 'feature_second': feature_second, 'feature_fps': feature_fps}
    
    # if not osp.exists('data/thumos14'):
    #     os.makedirs('data/thumos14')

    with open('data/thumos14/th14_img{}fps_info.json'.format(fps), 'w') as f:
        json.dump(result_dict, f)


if __name__ == '__main__':
    frame_dir = 'data/thumos14/thumos14_img10fps'
    gen_thumos14_frames_info(frame_dir, 10)