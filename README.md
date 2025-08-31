# Learning from Neighbors: Category Extrapolation for Long-Tail Learning

Shizhen Zhao, Xin Wen, Jiahui Liu, Chuofan Ma, Chunfeng Yuan, XiaoJuan Qi

This repository is the official PyTorch implementation of the paper [Learning from Neighbors](https://arxiv.org/abs/2410.15980).


## Environments

```shell
python == 3.7
pytorch >= 1.7.0
torchvision >= 0.8.1
timm == 0.3.2
tensorboardX >= 2.1
```
1. We recommand to install `PyTorch 1.7.0+`, `torchvision 0.8.1+` and `pytorch-image-models 0.3.2`.
2. If your PyTorch is 1.8.1+, a [fix](https://github.com/huggingface/pytorch-image-models/issues/420) is needed to work with timm.
3. See `requirements.txt` for detailed requirements. You don't have to be in strict agreement with it, just for reference.

## Data preparation

We adopt `torchvision.datasets.ImageFolder` to build our dataloaders. Hence, we resort all datasets (ImageNet-LT, iNat18, Places-LT) as follows:

```shell
/path/to/ImageNet-LT/
    train/
        class1/
            img1.jpeg
        class2/
            img2.jpeg
    val/
        class1/
            img3.jpeg
        class2/
            img4.jpeg
```
You can follow the `prepare.py` to construct your dataset.


## Usage

1. Please set the **DATA_PATH** and **WORK_PATH** in `util.trainer.py` Line 6-7.

2. Typically, make sure 4 or 8 GPUs and >12GB per GPU Memory are available.

The training and evaluation scripts are under the script folder, for example 

```python
# DINOV2
python script/DINOV2/finetune_dinov2_imagenet.py
# CLIP stage
python script/CLIP/finetune_clip_imagenet.py
# evaluate stage
python script/evaluate.py
```


## Citation
If you find our idea or code inspiring, please cite our paper:
```bibtex
@InProceedings{Zhao_2025_CVPR,
    author    = {Zhao, Shizhen and Wen, Xin and Liu, Jiahui and Ma, Chuofan and Yuan, Chunfeng and Qi, Xiaojuan},
    title     = {Learning from Neighbors: Category Extrapolation for Long-Tail Learning},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {30483-30492}
}
```
This code is partially based on [LiVT](https://github.com/XuZhengzhuo/LiVT), if you use our code, please also cite：
```bibtex
@InProceedings{Xu_2023_CVPR,
    author    = {Xu, Zhengzhuo and Liu, Ruikang and Yang, Shuo and Chai, Zenghao and Yuan, Chun},
    title     = {Learning Imbalanced Data With Vision Transformers},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {15793-15803}
}
```

## Acknowledgements
This project is highly based on [DeiT](https://github.com/facebookresearch/deit) and [MAE](https://github.com/facebookresearch/mae).
