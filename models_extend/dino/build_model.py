import os
import argparse
import json
from pathlib import Path

import torch
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms as pth_transforms
from torchvision import models as torchvision_models

from .utils import load_pretrained_weights
from timm.models.layers import trunc_normal_
from . import vision_transformer as vits

def build_model_dv1(num_classes=1000):

    model = vits.__dict__['vit_base_finetune'](patch_size=16, num_classes=num_classes)

    pretrained_weights = "/home/szzhao/LT_project/vit_LT/models_extend/dino/pretrained/dino_vitbase16_pretrain.pth"

    load_pretrained_weights(model, pretrained_weights)

    trunc_normal_(model.head.weight, std=2e-5)

    return model











