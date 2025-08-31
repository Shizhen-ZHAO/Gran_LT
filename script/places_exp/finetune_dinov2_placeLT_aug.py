import os
import sys
sys.path.append("..")
sys.path.append(os.getcwd())
from util.trainer import Trainer
from util.trainer import EXP_PATH, WORK_PATH

# def run_dinov2_placelt():
#     T = Trainer()
#     T.task = 'run_dinov2_placelt_aug_1'
#     T.note = 'run_dinov2_placelt_aug_1'
#     T.ckpt = './pretrained_models/dinov2_vitb14_pretrain.pth'
#
#     T.dataset = 'Place'
#     T.nb_classes = 365
#
#     T.epochs = 50
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
#     T.weight_decay = 0.1
#     T.adamW2 = 0.99
#
#     T.lr = 0.008
#     T.blr = 8.75e-6
#     T.layer_decay = 0.75
#     T.min_lr = 0.
#     T.warmup_epochs = 5
#
#     T.color_jitter = None
#     T.aa = 'rand-m9-mstd0.5-inc1'
#
#     T.reprob = 0.25
#     T.remode = 'pixel'
#     T.recount = 1
#     T.resplit = False
#     T.clswarm = 3
#
#
#     T.mixup = 0.1
#     T.cutmix = 0.1
#     T.cutmix_minmax = None
#     T.mixup_prob = 1.0
#     T.mixup_switch_prob = 0.5
#     T.mixup_mode = 'batch'
#
#     T.loss = 'Bal_CE'
#     T.bal_tau = 1.05
#     T.smoothing = 0.1
#
#     T.global_pool = True
#
#     T.seed = 0
#     T.prit = 1
#
#     T.num_workers = 16
#     T.master_port = 29502
#     T.lp = 0
#
#     T.clswarm = 3
#
#     T.mask_type = "no_fm_remove_normal_neg"
#     T.aug_data = "yes"
#     # T.gran_path = "/mnt/sda/szzhao/data_aug/places"
#     T.gran_path = "/mnt/sdc/zsz/aug_images/places_exp/exp_2"
#
#     T.finetune()

# def run_dinov2_placelt():
#     T = Trainer()
#     T.task = 'run_dinov2_placelt_aug_3'
#     T.note = 'run_dinov2_placelt_aug_3'
#     T.ckpt = './pretrained_models/dinov2_vitb14_pretrain.pth'
#
#     T.dataset = 'Place'
#     T.nb_classes = 365
#
#     T.epochs = 50
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
#     T.weight_decay = 0.1
#     T.adamW2 = 0.99
#
#     T.lr = 0.008
#     T.blr = 8.75e-6
#     T.layer_decay = 0.75
#     T.min_lr = 0.
#     T.warmup_epochs = 5
#
#     T.color_jitter = None
#     T.aa = 'rand-m9-mstd0.5-inc1'
#
#     T.reprob = 0.25
#     T.remode = 'pixel'
#     T.recount = 1
#     T.resplit = False
#     T.clswarm = 3
#
#
#     T.mixup = 0.1
#     T.cutmix = 0.1
#     T.cutmix_minmax = None
#     T.mixup_prob = 1.0
#     T.mixup_switch_prob = 0.5
#     T.mixup_mode = 'batch'
#
#     T.loss = 'Bal_CE'
#     T.bal_tau = 1.05
#     T.smoothing = 0.1
#
#     T.global_pool = True
#
#     T.seed = 0
#     T.prit = 1
#
#     T.num_workers = 16
#     T.master_port = 29553
#     T.lp = 0
#
#     T.clswarm = 3
#
#     T.mask_type = "no_fm_remove_normal_neg"
#     T.aug_data = "yes"
#     T.gran_path = "/mnt/sda/szzhao/data_aug/places"
#     # T.gran_path = "/mnt/sdc/zsz/aug_images/places_exp/exp_2"
#
#     T.finetune()
#
# run_dinov2_placelt()


# def run_dinov2_placelt():
#     T = Trainer()
#     T.task = 'run_dinov2_placelt_aug_5'
#     T.note = 'run_dinov2_placelt_aug_5'
#     T.ckpt = './pretrained_models/dinov2_vitb14_pretrain.pth'
#
#     T.dataset = 'Place'
#     T.nb_classes = 365
#
#     T.epochs = 50
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
#     T.weight_decay = 0.1
#     T.adamW2 = 0.99
#
#     T.lr = 0.008
#     T.blr = 8.75e-6
#     T.layer_decay = 0.75
#     T.min_lr = 0.
#     T.warmup_epochs = 5
#
#     T.color_jitter = None
#     T.aa = 'rand-m9-mstd0.5-inc1'
#
#     T.reprob = 0.25
#     T.remode = 'pixel'
#     T.recount = 1
#     T.resplit = False
#     T.clswarm = 3
#
#     T.mixup = 0.4
#     T.cutmix = 0.4
#     T.cutmix_minmax = None
#     T.mixup_prob = 1.0
#     T.mixup_switch_prob = 0.5
#     T.mixup_mode = 'batch'
#
#     T.loss = 'Bal_CE'
#     T.bal_tau = 1.05
#     T.smoothing = 0.1
#
#     T.global_pool = True
#
#     T.seed = 0
#     T.prit = 1
#
#     T.num_workers = 16
#     T.master_port = 29555
#     T.lp = 0
#
#     T.clswarm = 3
#
#     T.mask_type = "no_fm_remove_normal_neg"
#     T.aug_data = "yes"
#     T.gran_path = "/mnt/sda/szzhao/data_aug/places"
#     # T.gran_path = "/mnt/sdc/zsz/aug_images/places_exp/exp_2"
#
#     T.finetune()
#
# run_dinov2_placelt()


# def run_dinov2_placelt():
#     T = Trainer()
#     T.task = 'run_dinov2_placelt_aug_6'
#     T.note = 'run_dinov2_placelt_aug_6'
#     T.ckpt = './pretrained_models/dinov2_vitb14_pretrain.pth'
#
#     T.dataset = 'Place'
#     T.nb_classes = 365
#
#     T.epochs = 50
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
#     T.weight_decay = 0.1
#     T.adamW2 = 0.99
#
#     T.lr = 0.008
#     T.blr = 8.75e-6
#     T.layer_decay = 0.75
#     T.min_lr = 0.
#     T.warmup_epochs = 5
#
#     T.color_jitter = None
#     T.aa = 'rand-m9-mstd0.5-inc1'
#
#     T.reprob = 0.25
#     T.remode = 'pixel'
#     T.recount = 1
#     T.resplit = False
#     T.clswarm = 3
#
#     T.mixup = 0.6
#     T.cutmix = 0.6
#     T.cutmix_minmax = None
#     T.mixup_prob = 1.0
#     T.mixup_switch_prob = 0.5
#     T.mixup_mode = 'batch'
#
#     T.loss = 'Bal_CE'
#     T.bal_tau = 1.05
#     T.smoothing = 0.1
#
#     T.global_pool = True
#
#     T.seed = 0
#     T.prit = 1
#
#     T.num_workers = 16
#     T.master_port = 29556
#     T.lp = 0
#
#     T.clswarm = 3
#
#     T.mask_type = "no_fm_remove_normal_neg"
#     T.aug_data = "yes"
#     T.gran_path = "/mnt/sda/szzhao/data_aug/places"
#     # T.gran_path = "/mnt/sdc/zsz/aug_images/places_exp/exp_2"
#
#     T.finetune()
#
# run_dinov2_placelt()


# def run_dinov2_placelt():
#     T = Trainer()
#     T.task = 'run_dinov2_placelt_aug_7'
#     T.note = 'run_dinov2_placelt_aug_7'
#     T.ckpt = './pretrained_models/dinov2_vitb14_pretrain.pth'
#
#     T.dataset = 'Place'
#     T.nb_classes = 365
#
#     T.epochs = 50
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
#     T.weight_decay = 0.1
#     T.adamW2 = 0.99
#
#     T.lr = 0.008
#     T.blr = 8.75e-6
#     T.layer_decay = 0.75
#     T.min_lr = 0.
#     T.warmup_epochs = 5
#
#     T.color_jitter = None
#     T.aa = 'rand-m9-mstd0.5-inc1'
#
#     T.reprob = 0.25
#     T.remode = 'pixel'
#     T.recount = 1
#     T.resplit = False
#     T.clswarm = 3
#
#     T.mixup = 0.2
#     T.cutmix = 0.2
#     T.cutmix_minmax = None
#     T.mixup_prob = 1.0
#     T.mixup_switch_prob = 0.5
#     T.mixup_mode = 'batch'
#
#     T.loss = 'Bal_CE'
#     T.bal_tau = 1.05
#     T.smoothing = 0.1
#
#     T.global_pool = True
#
#     T.seed = 0
#     T.prit = 1
#
#     T.num_workers = 16
#     T.master_port = 29556
#     T.lp = 0
#
#     T.clswarm = 3
#
#     T.mask_type = "no_fm_remove_normal_neg"
#     T.aug_data = "yes"
#     T.gran_path = "/mnt/sda/szzhao/data_aug/places"
#     T.clswarm_scale = 30
#     # T.gran_path = "/mnt/sdc/zsz/aug_images/places_exp/exp_2"
#
#     T.finetune()
#
# run_dinov2_placelt()

def run_dinov2_placelt():
    T = Trainer()
    T.task = 'run_dinov2_placelt_aug_8'
    T.note = 'run_dinov2_placelt_aug_8'
    T.ckpt = './pretrained_models/dinov2_vitb14_pretrain.pth'

    T.dataset = 'Place'
    T.nb_classes = 365

    T.epochs = 50
    T.device = '4,5,6,7'

    T.batch = 128
    T.accum_iter = 1

    T.model = 'vit_base_patch14_dinov2'
    T.input_size = 224
    T.drop_path = 0.1

    T.clip_grad = None
    T.weight_decay = 0.1
    T.adamW2 = 0.99

    T.lr = 0.008
    T.blr = 1.75e-5
    T.layer_decay = 0.75
    T.min_lr = 0.
    T.warmup_epochs = 5

    T.color_jitter = None
    T.aa = 'rand-m9-mstd0.5-inc1'

    T.reprob = 0.25
    T.remode = 'pixel'
    T.recount = 1
    T.resplit = False
    T.clswarm = 3

    T.mixup = 0.2
    T.cutmix = 0.2
    T.cutmix_minmax = None
    T.mixup_prob = 1.0
    T.mixup_switch_prob = 0.5
    T.mixup_mode = 'batch'

    T.loss = 'Bal_CE'
    T.bal_tau = 1.05
    T.smoothing = 0.1

    T.global_pool = True

    T.seed = 0
    T.prit = 1

    T.num_workers = 16
    T.master_port = 29557
    T.lp = 0

    T.clswarm = 3

    T.mask_type = "no_fm_remove_normal_neg"
    T.aug_data = "yes"
    T.gran_path = "/mnt/sda/szzhao/data_aug/places"
    # T.clswarm_scale = 30
    # T.gran_path = "/mnt/sdc/zsz/aug_images/places_exp/exp_2"

    T.finetune()

run_dinov2_placelt()