import os
import sys
sys.path.append("..")
sys.path.append(os.getcwd())
from util.trainer import Trainer
from util.trainer import EXP_PATH, WORK_PATH


def eval_model():
    T = Trainer()
    # T.dataset = 'iNat18'
    T.dataset = 'ImageNet-LT'
    # T.dataset = 'Place'
    # T.resume = '/home/szzhao/LT_project/vit_LT/ckpt/run_dinov2adaptor_imagenetlt/ImageNet-LT/vit_base_patch14_dinov2/run_dinov2adaptor_imagenetlt/checkpoint.pth'
    # T.resume = '/home/szzhao/LT_project/vit_LT/ckpt/run_dinov2_inat224_CE_loss/iNat18/vit_base_patch14_dinov2/run_dinov2_inat224_CE_loss/checkpoint.pth'
    # T.resume = '/home/szzhao/LT_project/vit_LT/ckpt/run_dinov2_inat224_linear_probing/iNat18/vit_base_patch14_dinov2/run_dinov2_inat224_linear_probing/checkpoint.pth'
    # T.resume = '/home/szzhao/LT_project/vit_LT/ckpt/run_mae_imagenetlt_CE_loss_stage2/ImageNet-LT/vit_base_patch16/run_mae_imagenetlt_CE_loss_stage2/checkpoint.pth'
    # T.resume = '/home/szzhao/LT_project/vit_LT/ckpt/run_dinov2_inat224_granularity_related/iNat18/vit_base_patch14_dinov2/run_dinov2_inat224_granularity_related/checkpoint.pth'
    # T.resume = '/home/szzhao/LT_project/vit_LT/ckpt/run_dinov2_inat224_granularity_related_stage2/iNat18/vit_base_patch14_dinov2/run_dinov2_inat224_granularity_related_stage2/checkpoint.pth'
    # T.resume = '/home/szzhao/LT_project/vit_LT/ckpt/run_dinov2_inat224_granularity_unrelated_stage2/iNat18/vit_base_patch14_dinov2/run_dinov2_inat224_granularity_unrelated_stage2/checkpoint.pth'
    # T.resume = '/home/szzhao/LT_project/vit_LT/ckpt/run_dinov2_inat224_granularity_unrelated_stage2_con/iNat18/vit_base_patch14_dinov2/run_dinov2_inat224_granularity_unrelated_stage2_con/checkpoint.pth'
    # T.resume = '/home/szzhao/LT_project/vit_LT/ckpt/run_dinov2_inat224_granularity_related_stage2_con/iNat18/vit_base_patch14_dinov2/run_dinov2_inat224_granularity_related_stage2_con/checkpoint.pth'
    # T.resume = '/home/szzhao/LT_project/vit_LT/ckpt/run_dinov2_inat224_granularity_unrelated_v0/iNat18/vit_base_patch14_dinov2/run_dinov2_inat224_granularity_unrelated_v0/checkpoint.pth'
    # T.resume = '/home/szzhao/LT_project/vit_LT/ckpt/run_dinov2_inat224_granularity_confused_v0/iNat18/vit_base_patch14_dinov2/run_dinov2_inat224_granularity_confused_v0/checkpoint.pth'
    # T.resume = '/home/szzhao/LT_project/vit_LT/ckpt/run_dinov2_inat224_granularity_unrelated_v0_stage2_con/iNat18/vit_base_patch14_dinov2/run_dinov2_inat224_granularity_unrelated_v0_stage2_con/checkpoint.pth'
    # T.resume = '/home/szzhao/LT_project/vit_LT/ckpt/run_dinov2_imagenetlt_granularity_all_au_lr875/ImageNet-LT/vit_base_patch14_dinov2/run_dinov2_imagenetlt_granularity_all_au_lr875/checkpoint.pth'
    # T.resume = '/home/szzhao/LT_project/vit_LT/ckpt/run_dinov2_imagenetlt_granularity_all_au_lr875_mask_type_no_fm_remove_normal_neg_contrastive_3407/ImageNet-LT/vit_base_patch14_dinov2/run_dinov2_imagenetlt_granularity_all_au_lr875_mask_type_no_fm_remove_normal_neg_contrastive_3407/checkpoint.pth'
    # T.resume = '/home/szzhao/LT_project/vit_LT/ckpt/run_dinov2_inat224_granularity_related/iNat18/vit_base_patch14_dinov2/run_dinov2_inat224_granularity_related/checkpoint.pth'
    # T.resume = '/home/szzhao/LT_project/vit_LT/ckpt/run_clip_imagenetlt_aug/ImageNet-LT/vit_base_patch16_clip/run_clip_imagenetlt_aug/checkpoint.pth'
    # T.resume = "/home/szzhao/LT_project/vit_LT/ckpt/run_clip_placelt/Place/vit_base_patch16_clip/run_clip_placelt/checkpoint.pth"
    # T.resume = '/home/szzhao/LT_project/vit_LT/ckpt/run_clip_imagenetlt_smooth_loss_1/ImageNet-LT/vit_base_patch16_clip/run_clip_imagenetlt_smooth_loss_1/checkpoint.pth'
    # T.resume = '/home/szzhao/LT_project/vit_LT/ckpt/run_clip_placelt_aug/Place/vit_base_patch16_clip/run_clip_placelt_aug/checkpoint.pth'
    # T.resume = '/home/szzhao/LT_project/vit_LT/ckpt/run_scratch_place_aug/Place/vit_base_patch16/run_scratch_place_aug/checkpoint.pth'
    # T.resume = '/home/szzhao/LT_project/vit_LT/ckpt/run_scratch_place/Place/vit_base_patch16/run_scratch_place/checkpoint.pth'
    # T.resume = '/home/szzhao/LT_project/vit_LT/ckpt/run_scratch_imagenet_aug/ImageNet-LT/vit_base_patch16/run_scratch_imagenet_aug/checkpoint.pth'
    # T.resume = '/home/szzhao/LT_project/vit_LT/ckpt/run_resnet_placelt_aug_1/Place/resnet152/run_resnet_placelt_aug_1/checkpoint.pth'
    # T.resume = '/home/szzhao/LT_project/vit_LT/ckpt/run_resnet_placelt_3/Place/resnet152/run_resnet_placelt_3/checkpoint.pth'
    # T.resume = "/dataset/xinwen/work/LT_vit/ckpt/run_scratch_inat18/iNat18/vit_base_patch16/run_scratch_inat18/checkpoint.pth"
    # T.resume = "/dataset/xinwen/work/LT_vit/ckpt/run_scratch_inat18_2/iNat18/vit_base_patch16/run_scratch_inat18_2/checkpoint.pth"
    # T.resume = "/home/jiahliu/project008/ckpt/run_clip_inat18_clip_aug_1/iNat18/vit_base_patch16_clip/run_clip_inat18_clip_aug_1/checkpoint.pth"
    # T.resume = "/home/jiahliu/project008/ckpt/run_scratch_inat18_aug_2/iNat18/vit_base_patch16/run_scratch_inat18_aug_2/checkpoint.pth"
    # T.resume = "/home/szzhao/LT_project/vit_LT/ckpt/run_clip_imagenetlt_aug_9/ImageNet-LT/vit_base_patch16_clip/run_clip_imagenetlt_aug_9/checkpoint.pth"
    # T.resume = "/home/szzhao/LT_project/vit_LT/ckpt/run_dinov2_imagenetlt_aug_1/ImageNet-LT/vit_base_patch14_dinov2/run_dinov2_imagenetlt_aug_1/checkpoint.pth"
    # T.resume = "/home/szzhao/LT_project/vit_LT/ckpt/run_scratch_imagenet_aug_val_1/ImageNet-LT/vit_base_patch16/run_scratch_imagenet_aug_val_1/checkpoint.pth"
    # T.resume = "/home/szzhao/LT_project/vit_LT/ckpt/run_scratch_imagenet_aug_baseline/ImageNet-LT/vit_base_patch16/run_scratch_imagenet_aug_baseline/checkpoint.pth"
    # T.resume = "/home/szzhao/LT_project/vit_LT/ckpt/run_clip_imagenet1k/ImageNet-LT/vit_base_patch16_clip/run_clip_imagenet1k/checkpoint.pth"
    T.resume = "/home/szzhao/LT_project/vit_LT/ckpt/run_dinov2_imagenetlt_aug_baseline/ImageNet-LT/vit_base_patch14_dinov2/run_dinov2_imagenetlt_aug_baseline/checkpoint.pth"
    T.batch = 128
    T.device = '7'
    T.model = 'vit_base_patch14_dinov2'
    # T.model = 'vit_base_patch16'
    # T.model = 'resnet152'
    # T.model = 'resnet50'
    # T.model = "vit_base_patch16_clip"

    T.input_size = 224
    T.global_pool = True
    T.num_workers = 16
    T.master_port = 29625
    T.text_classifier=True
    T.aug_data = "yes"

    # T.gran_path = "/home/szzhao/LT_project/vit_LT/granularity_data/inat_18_sub/most_related"
    # T.gran_path = "/home/szzhao/LT_project/vit_LT/granularity_data/inat_18_sub/unrelated"
    # T.gran_path = "/home/szzhao/LT_project/vit_LT/granularity_data/ inat_18_v3/unrelated"
    # T.gran_path = "/home/szzhao/LT_project/vit_LT/granularity_data/inat_18_v3/related"
    # T.gran_path = "/home/szzhao/LT_project/vit_LT/granularity_data/inat_18_v3/unrelated_half"
    # T.gran_path = "/home/szzhao/LT_project/vit_LT/granularity_data/inat_18_v3/confused"
    # T.gran_path = "/home/szzhao/LT_project/vit_LT/granularity_data/inat_18_v3/related_prehalf"
    # T.gran_path = "/home/szzhao/LT_project/vit_LT/data/ImageNet-LT"
    # T.gran_path = "/mnt/sda/zsz/data/imagenetLT_auxiliary/s10000m2"
    # T.gran_path = "/mnt/sda/zsz/data/imagenetLT_auxiliary/all_au"
    # T.gran_path = "/mnt/sdc/zsz/aug_images/places_exp/exp_2"
    # T.gran_path = "./data/Place"
    # T.gran_path = "/home/szzhao/LT_project/vit_LT/data/Place"
    # T.gran_path = "/mnt/sdc/zsz/aug_images/places_exp/exp_2"
    T.gran_path = './data/ImageNet-LT'
    # T.gran_path = "/dataset/xinwen/inat18_new"
    # T.gran_path = "/mnt/sda/szzhao/data_aug/iNat18"
    # T.gran_path = "/mnt/sda/szzhao/data_aug/places"
    # T.gran_path = "/mnt/sda/szzhao/data/ImageNet-LT"
    # T.gran_path = "/mnt/sda/zsz/data/imagenetLT_auxiliary/all_au"

    T.evaluate()

eval_model()