# E2E-TAD
This repository holds the code for the following paper:
> [An Empirical Study of End-to-end Temporal Action Detection](https://arxiv.org/abs/2204.02932) <br/>
> [Xiaolong Liu](https://github.com/xlliu7), [Song Bai](https://songbai.site), [Xiang Bai](https://scholar.google.com/citations?user=UeltiQ4AAAAJ&hl=zh-CN) <br/>
> CVPR 2022.

This paper presents an empirical study of end-to-end temporal action detection (TAD). It 
- reveals the benefit of end-to-end training. We observe up to 11% performance improvement.
- studies the effect of a series of design choices in end-to-end TAD, including detection head, video encoder, spatial and temporal resolution of videos, frame sampling manner and multi-scale feature fusion.
- establishes a baseline detector. Built upon SlowFast and [TadTR][tadtr], it outperforms previous SOTA methods such as MUSES and AFSD with more than 4x faster speed, *using only RGB modality*. It can process *5076 frames per second* on a single TITAN Xp GPU. <br/>
![the baseline detector](figs/speed-accuracy.png)

<!-- We're currently refactoring the codebase to make it a generic framework for end-to-end temporal action detection. We hope to release the  -->
We hope that E2E-TAD can accelerate the research and applications of end-to-end temporal action detection. 

This code is an extended version of TadTR. It supports both video inputs and video feature inputs. If you want to run with video features, please refer to the instruction [here](https://github.com/xlliu7/TadTR/blob/master/README.md).

## TODOs
- [x] Inference code

- [ ] Training code

## Updates
Aug 14, 2022: The inference code for THUMOS14 dataset is released.

Apr, 2022: This repo is online.

Mar, 2022: Our paper is accepted by CVPR 2022.

## Results
### THUMOS14
|Config|Encoder|Head   |SR, TR |  AmAP |GPU Mem.|Training Speed|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|[link](configs/thumos14_e2e_slowfast_tadtr.yml)|SlowFast R50 4x16| TadTR|96, 10FPS| 54.2|7.6G|17 min/epoch|


### ActivityNet
|Config|Encoder| Head   |SR, TR|  AmAP |GPU Mem.|Train Speed|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|link|SlowFast R50 4x16| TadTR|96, 384|35.10|11G|62 (30)\* min/epoch|
|link|TSM R50|TadTR|96, 96|34.14|10G|30 (19) min/epoch|
<!-- |-|TSM R18|[TadTR][tadtr]|96, 96|33.42|3.6G|12 (8) min/epoch|[OneDrive]| -->

\* The values in the brackets are measured on RTX 3090, others on TITAN Xp.

SR: spatial (image) resolution. TR: temporal resolution, measured by the sampling frame rate on THUMOS14 and the number of sampled frames per video on ActivityNet.

## 0.Install 
### Requirements

* Linux or Windows. *Better with SSD*, because end-to-end has high IO demand.
  
* Python>=3.7

* CUDA>=9.2, GCC>=5.4
  
* PyTorch>=1.5.1, torchvision>=0.6.1 (following instructions [here](https://pytorch.org/))
  
* Other requirements
    ```bash
    pip install -r requirements.txt
    ```
### Compiling CUDA extensions
The RoIAlign operator is implemented with CUDA extension.
<!-- If your machine does have a NVIDIA GPU with CUDA support, you can run this step. Otherwise, please set `disable_cuda=True` in `opts.py`. -->
```bash
cd model/ops;

# If you have multiple installations of CUDA Toolkits, you'd better add a prefix
# CUDA_HOME=<your_cuda_toolkit_path> to specify the correct version. 
python setup.py build_ext --inplace
```

### Run a quick test
```
python demo.py --cfg configs/thumos14_e2e_slowfast_tadtr.yml
```

## 1.Data Preparation
### THUMOS14
Download video frames and annotation files from [[BaiduDrive](code: adTR)](https://pan.baidu.com/s/1MZtPjUSO_AlEqJmmhCFc3Q?pwd=adTR) or [[OneDrive]](https://husteducn-my.sharepoint.com/:f:/g/personal/liuxl_hust_edu_cn/Eglcf3femvhEpNl-5mmhDs4B0RiXiSX6UVOiOdUtG-VSTQ?e=XQm3jc).

Put all the following data under `data/thumos14` directory.

- Annotations: The annotations of action instances and the meta information of video files. Both are in JSON format (`th14_annotations_with_fps_duration.json` and `th14_img10fps_info.json`). The meta information file records the FPS and number of frame of each video. You can generated it by yourself using the script `tools/prepare_data.py`.
- Video Frames: We provide video frames extracted at 10fps. Extract the archive using `tar -xf thumos14_img10fps.tar`.
- Pre-trained Reference Model: Our pretrained model that uses SlowFast R50 encoder `thumos14_e2e_slowfast_tadtr_reference.pth`. This model corresponds to the config file `configs/thumos14_e2e_slowfast_tadtr.yml`. 

Note that our code does not depend on raw videos. You can download them mannually from the official site if you need them. We also provide a script that can download videos and extract frames. Please refer to `tools/extract_frames.py`.

## 2.Testing Pre-trained Models
Run
```
python main.py --cfg CFG_PATH --eval --resume CKPT_PATH
```
CFG_PATH is the path to the YAML-format config file that defines the experimental setting. For example, `configs/thumos14_e2e_slowfast_tadtr.yml`. CKPT_PATH is the path of the pre-trained model. Alternatively, you can execute the Shell script `bash scripts/test_reference_models_e2e.sh thumos14` for simplity.


## 3.Training by Yourself 
To be done. We are still checking the codebase. Plan to add official support of training in the next week.
<!-- (Preview version) -->

<!-- We include the training code. But it is still being tested. You can try it and report problems to me. -->

<!-- Run the following command
```
python main.py --cfg CFG_PATH
```

This codebase supports running on both single GPU or multiple GPUs. 
You may specify the GPU device ID (e.g., 0) to use by the adding the prefix `CUDA_VISIBLE_DEVICES=ID ` before the above command. To run on multiple GPUs, please refer to `scripts/run_parallel.sh`.

During training, our code will automatically perform testing every N epochs (N is the `test_interval` in opts.py). You can also monitor the training process with Tensorboard (need to set `cfg.tensorboard=True` in `opts.py`). The tensorboard record and the checkpoint will be saved at `output_dir` (can be modified in config file).

After training is done, you can also test your trained model by running
```
python main.py --cfg CFG_PATH --eval
```
It will automatically use the best model checkpoint. If you want to manually specify the model checkpoint, run
```
python main.py --cfg CFG_PATH --eval --resume CKPT_PATH
```

Note that the performance of the model trained by your own may be different from the reference model. The reason is that the gradient computation of the RoIAlign and Deformable Attention operators is not deterministic. Please refer to [this page](https://pytorch.org/docs/stable/notes/randomness.html) for details. -->

## Using Our Code on Your Dataset
Please refer to [docs/1_train_on_your_dataset.md](docs/1_train_on_your_dataset.md).

## Acknowledgement
The code is based on our previous project [TadTR][tadtr]. We also borrow some code from [DETR](https://github.com/facebookresearch/detr), [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR), [G-TAD](https://github.com/Frostinassiky/gtad), [TSM](https://github.com/mit-han-lab/temporal-shift-module) and [MMAction2](https://github.com/open-mmlab/mmaction2). Thanks for their great works.



## Citation
```
@inproceedings{liu2022an,
  title={An Empirical Study of End-to-end Temporal Action Detection},
  author={Liu, Xiaolong and Bai, Song and Bai, Xiang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={20010-20019},
  year={2022}
}
```

## Related Projects
- [TadTR][tadtr]: an early version of this project. TadTR is an efficient and flexible Transformer network for temporal action detection.


[tadtr]: https://github.com/xlliu7/TadTR
