import os
import sys
import time

from clip import load as clip_load

from .model import CLIP
from functools import partial
import math
import logging
from typing import Sequence, Tuple, Union, Callable

import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn.init import trunc_normal_
from timm.models.layers import trunc_normal_
# from clip import clip
import pickle

import clip

from .classes import CLASSES_INAT, CLASSES_Places, CLASSES, CUSTOM_TEMPLATES


class CLIP_Finetune(nn.Module):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, num_classes=1000, text_classifier=False, class_to_idx=None, dataset_name='', is_aug='no'):

        super().__init__()

        model, preprocess = clip_load('ViT-B/16', "cpu")
        self.visual = model.visual
        self.blocks = self.visual.transformer.resblocks

        # trunc_normal_(self.head.weight, std=2e-5)
        # is_aug = "yes"
        if dataset_name == 'ImageNet-LT':
            self.cls_names = CLASSES
            self.dataset_name = "imagenet"
            if is_aug == 'no':
                cls_weight_path = "/clip_text_embedding/imagenet1k.pkl"
            else:
                cls_weight_path = "/clip_text_embedding/classifier_weights_imagenetLT_aug.pkl"
        elif dataset_name == 'iNat18':
            if is_aug == 'no':
                cls_weight_path = "/clip_text_embedding/iNat18.pkl"
            else:
                cls_weight_path = "/clip_text_embedding/classifier_weights_iNat18_aug.pkl"
        elif dataset_name == 'Place':
            if is_aug == 'no':
                cls_weight_path = "/clip_text_embedding/place.pkl"
            else:
                cls_weight_path = "/clip_text_embedding/classifier_weights_placeLT_aug.pkl"

        self.text_classifier = text_classifier

        if text_classifier:
            # self.head = self.initial_classifier()
            # self.head = initial_classifier(self.cls_names)
            self.head = nn.Linear(self.visual.output_dim, num_classes, bias=False)
            root = os.path.abspath(__file__)

            save_path = root[:-len(root.split('/')[-1])] + cls_weight_path

            if os.path.exists(save_path):
                with open(save_path, 'rb') as file:
                    try:
                        text_embedding = pickle.load(file)['text_embedding'].cuda()
                    except:
                        with open(save_path, 'rb') as file:
                            text_embedding = torch.Tensor(pickle.load(file)).cuda()

            else:
                text_embedding = initial_classifier(self.cls_names)
                text_embedding_dict = {"text_embedding": text_embedding.cpu()}
                file = open(save_path, 'wb')
                pickle.dump(text_embedding_dict, file)
                print('asdas')
                sys.exit()

            for param_tensor in self.head.state_dict().keys():
                self.head.state_dict()[param_tensor].copy_(text_embedding)
            self.scale = 100.
        else:
            self.head = nn.Linear(self.visual.output_dim, num_classes)
            trunc_normal_(self.head.weight, std=2e-5)
            self.fc_norm = torch.nn.BatchNorm1d(self.visual.output_dim, affine=False, eps=1e-6)
            self.scale = 1.




    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def no_weight_decay(self):
        return {'positional_embedding'}


    def forward(self, image, **kwargs):

        image_features = self.visual(image.type(self.dtype)).float()

        if self.text_classifier:
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        else:
            image_features = self.fc_norm(image_features)

        Wstar = self.head.weight.T
        norm_Wstar = Wstar / torch.norm(Wstar, p=2, dim=0, keepdim=True)

        logits = self.scale * image_features @ norm_Wstar

        return logits, None


def build_model_clip(num_classes=1000, text_classifier=False, class_to_idx=None, dataset_name='', is_aug='no'):

    return CLIP_Finetune(num_classes=num_classes, text_classifier=text_classifier, class_to_idx=class_to_idx, dataset_name=dataset_name, is_aug=is_aug)


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        self.token_embedding = clip_model.token_embedding

    def forward(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x


def initial_classifier(cls_names):
    with torch.no_grad():
        classnames = cls_names
        templates = CUSTOM_TEMPLATES['ImageNet']

        # from clip import load as clip_load
        model, preprocess = clip_load('ViT-B/16', "cpu")
        torch.cuda.empty_cache()

        text_model = TextEncoder(model)

        text_model = torch.nn.DataParallel(text_model, device_ids = [0, 1, 2, 3])
        text_model.to(f'cuda:{text_model.device_ids[0]}')
        # self.text_model.to('cuda')
        # self.text_model.eval()

        for name, p in text_model.named_parameters():
            p.requires_grad = False

        texts = torch.cat([clip.tokenize(templates.format(c)) for c in classnames])
        # texts = texts.cuda()
        texts = texts.to(f'cuda:{text_model.device_ids[0]}')
        # texts = texts.to('cuda')
        zeroshot_weights = text_model(texts).float().detach()
        torch.cuda.empty_cache()

        texts = texts[:10]
        zeroshot_weights2 = text_model(texts[:10]).float().detach()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()

        texts.cpu()
        text_model.cpu()

        # del text_model
        # del texts

    return zeroshot_weights