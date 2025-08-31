# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import sys

from . import vision_transformer as vits
import models_extend.dinov2.utils.utils as dinov2_utils
from timm.models.layers import trunc_normal_
import torch

logger = logging.getLogger("dinov2")


def build_model_dinov2(args=None, only_teacher=False, img_size=224, num_classes=1000, description_token=None, class_to_idx=None, ckpt=None):
    arch = 'vit_base_finetune'
    img_size = 518
    patch_size = 14
    layerscale = 1e-05
    ffn_layer = 'mlp'
    block_chunks = 0
    qkv_bias = True
    proj_bias = True
    ffn_bias = True

    vit_kwargs = dict(
        num_classes=num_classes,
        img_size=img_size,
        patch_size=patch_size,
        init_values=layerscale,
        ffn_layer=ffn_layer,
        block_chunks=block_chunks,
        qkv_bias=qkv_bias,
        proj_bias=proj_bias,
        ffn_bias=ffn_bias,
        # description_token=description_token
        class_to_idx=class_to_idx
    )
    model = vits.__dict__[arch](**vit_kwargs)

    pretrained_weights = ckpt
    dinov2_utils.load_pretrained_weights(model, pretrained_weights, "teacher")

    trunc_normal_(model.head.weight, std=2e-5)

    if 'ckpt' not in pretrained_weights:
        trunc_normal_(model.head.weight, std=2e-5)

    # pretrained_weights = "/home/szzhao/LT_project/vit_LT/ckpt/run_dinov2adaptor_imagenetlt/ImageNet-LT/vit_base_patch14_dinov2/run_dinov2adaptor_imagenetlt/checkpoint.pth"
    # dinov2_utils.load_pretrained_weights(model, pretrained_weights, "teacher")

    return model





# def build_model_from_cfg(cfg, only_teacher=False):
#     return build_model(cfg.student, only_teacher=only_teacher, img_size=cfg.crops.global_crops_size)


