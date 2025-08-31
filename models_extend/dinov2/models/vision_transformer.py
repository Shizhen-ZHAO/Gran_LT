# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import sys
# References:
#   https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py
# from functools import partial
from functools import partial
import math
import logging
from typing import Sequence, Tuple, Union, Callable
import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn.init import trunc_normal_

from models_extend.dinov2.layers import Mlp, PatchEmbed, SwiGLUFFNFused, MemEffAttention, NestedTensorBlock as Block
from clip import clip
import os
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger("dinov2")


def named_apply(fn: Callable, module: nn.Module, name="", depth_first=True, include_root=False) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


class BlockChunk(nn.ModuleList):
    def forward(self, x):
        for b in self:
            x = b(x)
        return x

class DinoVisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        drop_path_rate=0.0,
        drop_path_uniform=False,
        init_values=None,  # for layerscale: None or 0 => no layerscale
        embed_layer=PatchEmbed,
        act_layer=nn.GELU,
        block_fn=Block,
        ffn_layer="mlp",
        block_chunks=1,
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            proj_bias (bool): enable bias for proj in attn if True
            ffn_bias (bool): enable bias for ffn if True
            drop_path_rate (float): stochastic depth rate
            drop_path_uniform (bool): apply uniform drop rate across blocks
            weight_init (str): weight init scheme
            init_values (float): layer-scale init values
            embed_layer (nn.Module): patch embedding layer
            act_layer (nn.Module): MLP activation layer
            block_fn (nn.Module): transformer block class
            ffn_layer (str): "mlp", "swiglu", "swiglufused" or "identity"
            block_chunks: (int) split block sequence into block_chunks units for FSDP wrap
        """
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))

        if drop_path_uniform is True:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        if ffn_layer == "mlp":
            logger.info("using MLP layer as FFN")
            ffn_layer = Mlp
        elif ffn_layer == "swiglufused" or ffn_layer == "swiglu":
            logger.info("using SwiGLU layer as FFN")
            ffn_layer = SwiGLUFFNFused
        elif ffn_layer == "identity":
            logger.info("using Identity layer as FFN")

            def f(*args, **kwargs):
                return nn.Identity()

            ffn_layer = f
        else:
            raise NotImplementedError

        blocks_list = [
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                ffn_layer=ffn_layer,
                init_values=init_values,
            )
            for i in range(depth)
        ]
        if block_chunks > 0:
            self.chunked_blocks = True
            chunked_blocks = []
            chunksize = depth // block_chunks
            for i in range(0, depth, chunksize):
                # this is to keep the block index consistent if we chunk the block list
                chunked_blocks.append([nn.Identity()] * i + blocks_list[i : i + chunksize])
            self.blocks = nn.ModuleList([BlockChunk(p) for p in chunked_blocks])
        else:
            self.chunked_blocks = False
            self.blocks = nn.ModuleList(blocks_list)

        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))

        self.init_weights()

    def init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        named_apply(init_weights_vit_timm, self)

    def interpolate_pos_encoding(self, x, w, h):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode="bicubic",
        )

        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    def prepare_tokens_with_masks(self, x, masks=None):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)

        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)

        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)

        return x

    def forward_features_list(self, x_list, masks_list):
        x = [self.prepare_tokens_with_masks(x, masks) for x, masks in zip(x_list, masks_list)]
        for blk in self.blocks:
            x = blk(x)

        all_x = x
        output = []
        for x, masks in zip(all_x, masks_list):
            x_norm = self.norm(x)
            output.append(
                {
                    "x_norm_clstoken": x_norm[:, 0],
                    "x_norm_patchtokens": x_norm[:, 1:],
                    "x_prenorm": x,
                    "masks": masks,
                }
            )
        return output

    def forward_features(self, x, masks=None):
        if isinstance(x, list):
            # print(1)
            return self.forward_features_list(x, masks)

        # print(2)
        x = self.prepare_tokens_with_masks(x, masks)

        for blk in self.blocks:
            x = blk(x)

        x_norm = self.norm(x)
        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_patchtokens": x_norm[:, 1:],
            "x_prenorm": x,
            "masks": masks,
        }

    def _get_intermediate_layers_not_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        # If n is an int, take the n last blocks. If it's a list, take them
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in blocks_to_take:
                output.append(x)
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def _get_intermediate_layers_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        output, i, total_block_len = [], 0, len(self.blocks[-1])
        # If n is an int, take the n last blocks. If it's a list, take them
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for block_chunk in self.blocks:
            for blk in block_chunk[i:]:  # Passing the nn.Identity()
                x = blk(x)
                if i in blocks_to_take:
                    output.append(x)
                i += 1
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence] = 1,  # Layers or n last layers to take
        reshape: bool = False,
        return_class_token: bool = False,
        norm=True,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        if self.chunked_blocks:
            outputs = self._get_intermediate_layers_chunked(x, n)
        else:
            outputs = self._get_intermediate_layers_not_chunked(x, n)
        if norm:
            outputs = [self.norm(out) for out in outputs]
        class_tokens = [out[:, 0] for out in outputs]
        outputs = [out[:, 1:] for out in outputs]
        if reshape:
            B, _, w, h = x.shape
            outputs = [
                out.reshape(B, w // self.patch_size, h // self.patch_size, -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]
        if return_class_token:
            return tuple(zip(outputs, class_tokens))
        return tuple(outputs)

    def forward(self, *args, is_training=False, **kwargs):
        ret = self.forward_features(*args, **kwargs)


        if is_training:
            return ret
        else:
            return self.head(ret["x_norm_clstoken"])


def init_weights_vit_timm(module: nn.Module, name: str = ""):
    """ViT weight initialization, original timm impl (for reproducibility)"""
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def vit_small(patch_size=16, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        **kwargs,
    )
    return model


def vit_base(patch_size=16, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        **kwargs,
    )
    return model


def vit_large(patch_size=16, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        **kwargs,
    )
    return model


def vit_giant2(patch_size=16, **kwargs):
    """
    Close to ViT-giant, with embed-dim 1536 and 24 heads => embed-dim per head 64
    """
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1536,
        depth=40,
        num_heads=24,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        **kwargs,
    )
    return model


class DinoVisionTransformer_Finetune(DinoVisionTransformer):
    """ Vision Transformer with support for global average pooling
    """

    def __init__(self, global_pool=True, language=True, class_to_idx=None, num_classes=1000, **kwargs):

        super(DinoVisionTransformer_Finetune, self).__init__(**kwargs)

        self.head = nn.Linear(self.embed_dim, num_classes, bias=True)

        # self.head = nn.utils.weight_norm(nn.Linear(self.embed_dim, num_classes, bias=False))
        # self.head.weight_g.data.fill_(1)
        # self.head.weight_g.requires_grad = False

        self.global_pool = global_pool

        if self.global_pool:
            # norm_layer = partial(nn.LayerNorm, eps=1e-6)
            # self.fc_norm = norm_layer(self.embed_dim)
            self.fc_norm = torch.nn.BatchNorm1d(self.embed_dim, affine=False, eps=1e-6)

            del self.norm  # remove the original norm

        self.init_weights()

    def forward_features_list(self, x_list, masks_list):
        x = [self.prepare_tokens_with_masks(x, masks) for x, masks in zip(x_list, masks_list)]
        for blk in self.blocks:
            x = blk(x)

        all_x = x
        output = []
        for x, masks in zip(all_x, masks_list):

            if self.global_pool:
                x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
                outcome = self.fc_norm(x)
            else:
                x = self.norm(x)
                outcome = x[:, 0]

            output.append(
                {
                    # "x_norm_clstoken": x_norm[:, 0],
                    "x_norm": outcome,
                    # "x_prenorm": x,
                    # "masks": masks,
                }
            )
        return output

    def forward_features(self, x, masks=None):
        if isinstance(x, list):
            return self.forward_features_list(x, masks)

        x = self.prepare_tokens_with_masks(x, masks)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:


            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
            # print("a")
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome, x[:, 1:]

    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward(self, *args, is_training=False, digital_targets=None, detach_aug='no',normal_mask=None, **kwargs):
        ret, spatial_feat = self.forward_features(*args, **kwargs)
        # print(digital_targets.shape)
        # print(1)
        # print()
        # if detach_aug == 'detach_aug':
            # digital_targets = torch.where(digital_targets >= 1000)[0]
        # batch_mask = normal_mask[digital_targets]
        # mask_index = torch.where(batch_mask == 0)[0]
        # ret[mask_index] = ret[mask_index].detach()


        return self.head(ret), ret


# class DinoVisionTransformer_TextFinetune(DinoVisionTransformer):
#     """ Vision Transformer with support for global average pooling
#     """
#     def __init__(self, global_pool=True, language=True, num_classes=1000, description_token=None, **kwargs):
#
#         super(DinoVisionTransformer_TextFinetune, self).__init__(**kwargs)
#
#         self.head = nn.Linear(self.embed_dim, num_classes)
#
#         self.global_pool = global_pool
#         if self.global_pool:
#             # norm_layer = partial(nn.LayerNorm, eps=1e-6)
#             # self.fc_norm = norm_layer(self.embed_dim)
#             self.fc_norm = torch.nn.BatchNorm1d(self.embed_dim, affine=False, eps=1e-6)
#
#             del self.norm  # remove the original norm
#
#         self.init_weights()
#
#         self.language = language
#
#         self.descriptions_tokens = torch.cat([description_token[i] for i in range(1000)], dim=0)
#         self.descriptions_num = torch.cat([torch.tensor(len(description_token[i])).unsqueeze(0) for i in range(1000)])
#
#         self.descriptions_num = torch.cat([torch.tensor(0).unsqueeze(0), self.descriptions_num])
#
#         if self.language:
#             self.num_language_token = torch.max(self.descriptions_num)
#             self.language_token = nn.Parameter(torch.zeros(1, self.num_language_token, self.embed_dim))
#             self.language_token.data.copy_(self.cls_token.data.detach().expand(-1, self.num_language_token, -1))
#             self.language_pos_embed = nn.Parameter(torch.zeros(1, self.num_language_token, self.embed_dim))
#
#             # nn.init.normal_(self.language_token, std=1e-6)
#             trunc_normal_(self.language_pos_embed, std=1e-6)
#
#             self.language_pos_embed.data.copy_(self.pos_embed.float().data[:, 0].unsqueeze(0).detach().expand(-1, self.num_language_token, -1) + self.language_pos_embed.data)
#
#             self.adaptor = nn.Linear(self.embed_dim, 512)
#
#             trunc_normal_(self.adaptor.weight, std=1e-6)
#
#         self.text_encoder = TextEncoder(load_clip_to_cpu())
#
#         self.descriptions_num = self.descriptions_num.cumsum(dim=0)
#
#         # print(self.descriptions_token.shape)
#         self.compositional_classifier = None
#
#         # self.descriptions_logits_scale = nn.Parameter(torch.tensor(1.0))
#
#
#     def prepare_tokens_with_masks(self, x, masks=None):
#         B, nc, w, h = x.shape
#         x = self.patch_embed(x)
#
#         if masks is not None:
#             x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)
#
#         x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x, self.language_token.expand(x.shape[0], -1, -1)), dim=1)
#
#         x = x + self.interpolate_pos_encoding(x, w, h)
#
#         return x
#
#     def generate_classifier(self):
#         self.compositional_classifier = self.text_encoder(self.descriptions_tokens.cuda())
#         self.compositional_classifier = self.compositional_classifier / self.compositional_classifier.norm(dim=-1, keepdim=True)
#         self.compositional_classifier = self.compositional_classifier.permute(1,0)
#
#     def generate_language_logits(self, language_embeddings):
#
#         # adapted_embeddings = self.adaptor(language_embeddings)
#         #
#         # batch_size, token_num, embedding_num = adapted_embeddings.shape
#         #
#         # adapted_embeddings = adapted_embeddings.reshape(-1, embedding_num)
#         #
#         # adapted_embeddings = adapted_embeddings / adapted_embeddings.norm(dim=-1, keepdim=True)
#         #
#         # similarity = adapted_embeddings.float() @ self.compositional_classifier.float()
#         #
#         # similarity = similarity.reshape(batch_size, token_num, -1)
#         #
#         # arrange_similarity = []
#         #
#         # for batch_index in range(similarity.shape[0]):
#         #     one_similarity = similarity[batch_index].T
#         #
#         #     minus_one_similarity = -one_similarity
#         #     minus_one_similarity = minus_one_similarity.detach().cpu().numpy()
#         #
#         #     # one_similarity = torch.cat([torch.mean(one_similarity[self.descriptions_num[i]:self.descriptions_num[i + 1], :][np.arange(0,self.descriptions_num[i + 1]-self.descriptions_num[i],1), linear_sum_assignment(minus_one_similarity[self.descriptions_num[i]:self.descriptions_num[i + 1], :])[1]]).unsqueeze(0) for i in range(1000)], dim=0).unsqueeze(0)
#         #     # for i in range(1000):
#         #     #     linear_sum_assignment(minus_one_similarity[self.descriptions_num[i]:self.descriptions_num[i + 1], :])
#         #
#         #     one_similarity = torch.rand(1, 1000).cuda()
#         #     arrange_similarity.append(one_similarity)
#         #
#         # one_similarity = torch.rand(128, 1000).cuda()
#         #
#         # # arrange_similarity = torch.cat(arrange_similarity, dim=0)
#
#         return  torch.rand(128, 1000).cuda()
#
#     def interpolate_pos_encoding(self, x, w, h):
#         previous_dtype = x.dtype
#         npatch = x.shape[1] - 1
#         N = self.pos_embed.shape[1] - 1
#         if npatch == N and w == h:
#             return self.pos_embed
#         pos_embed = self.pos_embed.float()
#         class_pos_embed = pos_embed[:, 0]
#         patch_pos_embed = pos_embed[:, 1:]
#         dim = x.shape[-1]
#         w0 = w // self.patch_size
#         h0 = h // self.patch_size
#         # we add a small number to avoid floating point error in the interpolation
#         # see discussion at https://github.com/facebookresearch/dino/issues/8
#         w0, h0 = w0 + 0.1, h0 + 0.1
#
#         patch_pos_embed = nn.functional.interpolate(
#             patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
#             scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
#             mode="bicubic",
#         )
#         assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
#         patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
#
#         return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed, self.language_pos_embed), dim=1).to(previous_dtype)
#
#     def forward_features_list(self, x_list, masks_list):
#         x = [self.prepare_tokens_with_masks(x, masks) for x, masks in zip(x_list, masks_list)]
#         for blk in self.blocks:
#             x = blk(x)
#
#         all_x = x
#         output = []
#         for x, masks in zip(all_x, masks_list):
#
#             if self.global_pool:
#                 x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
#                 outcome = self.fc_norm(x)
#             else:
#                 x = self.norm(x)
#                 outcome = x[:, 0]
#
#             output.append(
#                 {
#                     # "x_norm_clstoken": x_norm[:, 0],
#                     "x_norm": outcome,
#                     # "x_prenorm": x,
#                     # "masks": masks,
#                 }
#             )
#         return output
#
#     def forward_features(self, x, masks=None, text=None):
#         if isinstance(x, list):
#             return self.forward_features_list(x, masks)
#
#         x = self.prepare_tokens_with_masks(x, masks)
#
#         for blk in self.blocks:
#             x = blk(x)
#
#         if self.global_pool:
#             x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
#             outcome = self.fc_norm(x)
#             # print("a")
#         else:
#             x = self.norm(x)
#             outcome = x[:, 0]
#
#         return outcome, x[:, -self.num_language_token:]
#
#
#     def no_weight_decay(self):
#         return {'pos_embed', 'cls_token', 'language_token', 'language_pos_embed'}
#
#     def forward(self, *args, is_training=False, text=[], **kwargs):
#
#         if self.compositional_classifier is None:
#             self.generate_classifier()
#         # print(self.compositional_classifier.shape)
#
#         ret, language_output = self.forward_features(*args, **kwargs)
#
#         language_logits = self.generate_language_logits(language_output)
#         #
#         # if text != []:
#         #     return language_logits, self.text_encoder(text)
#         # else:
#         #     return language_logits, []
#
#
#         if text != []:
#             return self.head(ret), self.text_encoder(text)
#         else:
#             return self.head(ret), []

# class DinoVisionTransformer_Finetune(DinoVisionTransformer):
#     """ Vision Transformer with support for global average pooling
#     """
#
#     def __init__(self, global_pool=True, num_classes=1000, class_to_idx=None,**kwargs):
#
#         super(DinoVisionTransformer_Finetune, self).__init__(**kwargs)
#
#         # self.head = nn.Linear(self.embed_dim, num_classes)
#
#         self.global_pool = global_pool
#         # if self.global_pool:
#             # norm_layer = partial(nn.LayerNorm, eps=1e-6)
#             # self.fc_norm = norm_layer(self.embed_dim)
#         self.fc_norm = torch.nn.BatchNorm1d(self.embed_dim, affine=False, eps=1e-6)
#
#         del self.norm  # remove the original norm
#
#         self.init_weights()
#
#         self.head = nn.Linear(512, num_classes, bias=False)
#
#         weights = self.initial_head(num_classes, class_to_idx)
#
#         for param_tensor in self.head.state_dict().keys():
#             self.head.state_dict()[param_tensor].copy_(weights)
#
#         # print(weights)
#         self.adaptor = nn.Sequential(nn.Linear(self.embed_dim, 512), nn.ReLU(), nn.BatchNorm1d(512, affine=False, eps=1e-6), nn.Linear(512, 512), nn.ReLU(), nn.BatchNorm1d(512, affine=False, eps=1e-6), nn.Linear(512, 512))
#
#         for layer in self.adaptor:
#             if isinstance(layer, nn.Linear):  # 判断是否是线性层
#                 trunc_normal_(layer.weight, std=1e-6)
#
#     def initial_head(self, num_classes, class_to_idx):
#         import pickle
#         from models_extend.CLIP.clip.classes import CLASSES_INAT, CLASSES_Places, CLASSES, CUSTOM_TEMPLATES
#         from models_extend.CLIP.clip.clip import load as clip_load
#
#         if num_classes == 1000:
#             self.cls_names = CLASSES
#             self.dataset_name = "imagenet"
#         elif num_classes == 365:
#             self.cls_names = CLASSES_Places
#             self.dataset_name = "place"
#         elif num_classes > 8000:
#             self.cls_names = CLASSES_INAT
#             self.dataset_name = "inat"
#
#         new_names = [""] * num_classes
#         for k, v in class_to_idx.items():
#             original_id = int(k)
#             new_id = int(v)
#             new_names[new_id] = self.cls_names[original_id]
#         self.cls_names = new_names
#
#         save_root = "/home/szzhao/LT_project/vit_LT/models_extend/CLIP/clip/clip_text_embedding"
#         save_path = os.path.join(save_root, self.dataset_name + '.pkl')
#
#         with open(save_path, 'rb') as file:
#             text_embedding = pickle.load(file)['text_embedding'].cuda()
#
#         return text_embedding
#
#
#     def forward_features_list(self, x_list, masks_list):
#         x = [self.prepare_tokens_with_masks(x, masks) for x, masks in zip(x_list, masks_list)]
#         for blk in self.blocks:
#             x = blk(x)
#
#         all_x = x
#         output = []
#         for x, masks in zip(all_x, masks_list):
#
#             if self.global_pool:
#                 x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
#                 outcome = self.fc_norm(x)
#             else:
#                 x = self.fc_norm(x)
#                 outcome = x[:, 0]
#
#             output.append(
#                 {
#                     # "x_norm_clstoken": x_norm[:, 0],
#                     "x_norm": outcome,
#                     # "x_prenorm": x,
#                     # "masks": masks,
#                 }
#             )
#         return output
#
#     def forward_features(self, x, masks=None):
#         if isinstance(x, list):
#             return self.forward_features_list(x, masks)
#
#         x = self.prepare_tokens_with_masks(x, masks)
#
#         for blk in self.blocks:
#             x = blk(x)
#
#         if self.global_pool:
#             x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
#             outcome = self.fc_norm(x)
#             # print("a")
#         else:
#             # x = self.fc_norm(x)
#             outcome = self.fc_norm(x[:, 0])
#
#         return outcome
#
#     def no_weight_decay(self):
#         return {'pos_embed', 'cls_token'}
#
#     def forward(self, *args, is_training=False, **kwargs):
#         ret = self.forward_features(*args, **kwargs)
#         ret = self.adaptor(ret)
#         # ret = ret / ret.norm(dim=-1, keepdim=True)
#
#         Wstar = self.head.weight.T
#         norm_Wstar = Wstar / torch.norm(Wstar, p=2, dim=0, keepdim=True)
#
#         logits = ret @ norm_Wstar
#
#         return logits, ret


def vit_base_finetune(patch_size=16, **kwargs):
    model = DinoVisionTransformer_Finetune(
        global_pool=False,
        # num_classes=1000,
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        **kwargs,
    )
    return model


# def vit_tiny_patch16(**kwargs):
#     model = VisionTransformer(
#         patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     return model

# def vit_small_patch16(**kwargs):
#     model = VisionTransformer(
#         patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     return model

# def vit_base_patch16(**kwargs):
#     model = VisionTransformer(
#         patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     return model
#
# def vit_large_patch16(**kwargs):
#     model = VisionTransformer(
#         patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     return model
#
# def vit_huge_patch14(**kwargs):
#     model = VisionTransformer(
#         patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     return model




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

def load_clip_to_cpu(visual_backbone='ViT-B/16'):
    backbone_name = visual_backbone
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url, os.path.expanduser("~/.cache/clip"))

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model