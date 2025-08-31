import os
import sys
sys.path.append("..")
sys.path.append(os.getcwd())
from util.trainer import Trainer
from util.trainer import EXP_PATH, WORK_PATH

# def run_dinov2_imagenetlt():
#     T = Trainer()
#     T.task = 'run_dinov2_imagenetlt_granularity_all_au_lr875_mask_type_no_fm_2_remove_normal_neg_back_contrastive'
#     T.note = 'run_dinov2_imagenetlt_granularity_all_au_lr875_mask_type_no_fm_2_remove_normal_neg_back_contrastive'
#     T.ckpt = '/home/szzhao/pretrained_models/dinov2_vitb14_pretrain.pth'
#
#     T.dataset = 'ImageNet-LT'
#     T.nb_classes = 1000
#
#     T.epochs = 100
#     T.device = '4,5,6,7'
#
#     T.batch = 128
#     T.accum_iter = 1
#
#     T.model = 'vit_base_patch14_dinov2'
#     T.input_size = 224
#     T.drop_path = 0.1
#
#     T.clip_grad = None
#     T.weight_decay = 0.05
#     T.adamW2 = 0.99
#
#     T.lr = 0.008
#     T.blr = 8.75e-6
#     T.layer_decay = 0.75
#     T.min_lr = 0.
#     T.warmup_epochs = 10
#     T.clswarm = 3
#
#     T.color_jitter = None
#     T.aa = 'rand-m9-mstd0.5-inc1'
#
#     T.reprob = 0.25
#     T.remode = 'pixel'
#     T.recount = 1
#     T.resplit = False
#
#     T.mixup = 0.8
#     T.cutmix = 1.0
#     T.cutmix_minmax = None
#     T.mixup_prob = 1.0
#     T.mixup_switch_prob = 0.5
#     T.mixup_mode = 'batch'
#
#     T.loss = 'Bal_CE'
#     T.bal_tau = 1.0
#     T.smoothing = 0.1
#
#     T.global_pool = True
#
#     T.seed = 0
#     T.prit = 1
#
#     T.lp = 0
#     # T.gran_path = "/mnt/sda/zsz/data/imagenetLT_exp/ImageNet-LT-cluster"
#     T.gran_path = "/mnt/sda/zsz/data/imagenetLT_auxiliary/all_au"
#
#     T.num_workers = 16
#     T.master_port = 29520
#
#     T.mask_type = "no_fm_2_remove_normal_neg_back"
#
#     T.finetune()
#
# run_dinov2_imagenetlt()



def run_dinov2_imagenetlt():
    T = Trainer()
    T.task = 'run_dinov2_imagenetlt_granularity_all_au_lr875_mask_type_no_fm_remove_normal_neg_contrastive_adjust_sample_num'
    T.note = 'run_dinov2_imagenetlt_granularity_all_au_lr875_mask_type_no_fm_remove_normal_neg_contrastive_adjust_sample_num'
    T.ckpt = '/home/szzhao/pretrained_models/dinov2_vitb14_pretrain.pth'

    T.dataset = 'ImageNet-LT'
    T.nb_classes = 1000

    T.epochs = 100
    T.device = '4,5,6,7'

    T.batch = 128
    T.accum_iter = 1

    T.model = 'vit_base_patch14_dinov2'
    T.input_size = 224
    T.drop_path = 0.1

    T.clip_grad = None
    T.weight_decay = 0.05
    T.adamW2 = 0.99

    T.lr = 0.008
    T.blr = 8.75e-6
    T.layer_decay = 0.75
    T.min_lr = 0.
    T.warmup_epochs = 10
    T.clswarm = 3

    T.color_jitter = None
    T.aa = 'rand-m9-mstd0.5-inc1'

    T.reprob = 0.25
    T.remode = 'pixel'
    T.recount = 1
    T.resplit = False

    T.mixup = 0.8
    T.cutmix = 1.0
    T.cutmix_minmax = None
    T.mixup_prob = 1.0
    T.mixup_switch_prob = 0.5
    T.mixup_mode = 'batch'

    T.loss = 'Bal_CE'
    T.bal_tau = 1.0
    T.smoothing = 0.1

    T.global_pool = True

    T.seed = 0
    T.prit = 1

    T.lp = 0
    # T.gran_path = "/mnt/sda/zsz/data/imagenetLT_exp/ImageNet-LT-cluster"
    T.gran_path = "/mnt/sda/zsz/data/imagenetLT_auxiliary/all_au"

    T.num_workers = 16
    T.master_port = 29525

    T.mask_type = "no_fm_remove_normal_neg"

    T.finetune()

run_dinov2_imagenetlt()