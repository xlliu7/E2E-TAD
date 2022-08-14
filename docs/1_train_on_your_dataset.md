# Train and Evaluate E2E-TAD on Your Dataset

## 1. Prepare data
- ActivityNet-style annotations:
Our dataloader supports any dataset as soon as the annotation file has the same format as ActivityNet. See the example below. The path of this annotation file is denoted as `ANNO_PATH`.

```JSON
{
   "database": {
      "video_id": {
        "duration" : 12,
        "annotations": [
            {
                "class": "Futsal",
                "segment": [2.0, 18.0]
            }
        ]
      },
      "video_id2": {

      }
   }
}

```

- Video frames: Please refer to `tools/extract_frames.py` to extract video frames for your dataset. The root path of frames is denoted as `FRAME_PATH`. You should choose a proper FPS. If your dataset is similar to THUMOS14, you may extract frames at around 10 fps. If it is similar to ActivityNet, you may sample fixed number of frames from each video.

- Extra annotation file: Please refer to `tools/prepare_data.py` to generate a file that records the FPS and number of frame of each video. The path of this file is denoted as `FT_INFO_PATH`.

After these steps, please add the FRAME_PATH and FT_INFO_PATH info in `datasets/path.yml` for your dataset.
```
YOUR_DATASET:
  ann_file: ANNO_PATH
  img:     
    local_path: FRAME_PATH
    ft_info_file: FT_INFO_PATH
```

## 2. Modify code
- models/tadtr.py: modify the `build` function to specify the number of classes of your dataset.
- datasets/data_utils: modify the `get_dataset_info` function.
- datasets/tad_eval.py: modify line 66-72.
- engine.py: modify line 110.

## 3. Write a config file
Please refer to the existing config files.
You need to set some parameters. For example, 
- slice_len: if the videos are long and the actions are short, you may need to cut videos into slices (windows). The slice_len should be set to a value such that most actions are shorter than the corresponding duration. (slice_len = slice_duration * fps)
- the number of queries: it should be set to a value that is slightly larger than the maximum number of actions per video.

## 4. Training and evluation
Training and evaluation process will be the same as THUMOS14.

