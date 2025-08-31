import os
import sys
sys.path.append("..")
sys.path.append(os.getcwd())
from util.trainer import Trainer
from util.trainer import EXP_PATH, WORK_PATH

def run_placelt():
    T = Trainer()
    T.task = 'run_mae_place_CE_loss_stage1'
    T.note = 'run_mae_place_CE_loss_stage1'
    T.ckpt = './pretrained_models/imagenet1k_vit_base.pth'

    T.dataset = 'Place'
    T.nb_classes = 365

    T.epochs = 30
    T.device = '0,1,2,3'

    T.batch = 64
    T.accum_iter = 4

    T.model = 'vit_base_patch16'
    T.input_size = 224
    T.drop_path = 0.1

    T.clip_grad = None
    T.weight_decay = 0.05
    T.adamW2 = 0.99

    T.lr = 0.008
    T.blr = 1e-3
    T.layer_decay = 0.75
    T.min_lr = 0
    T.warmup_epochs = 5

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

    T.loss = 'CE'
    T.bal_tau = 1.05
    T.smoothing = 0.1

    T.global_pool = True

    T.seed = 0
    T.prit = 1

    T.num_workers = 16
    T.master_port = 29503

    T.clswarm = 3
    T.lp = 0

    T.finetune()

run_placelt()
#
# def run_placelt_accum_iter8():
#     T = Trainer()
#     T.task = 'run_placelt_accum_iter8'
#     T.note = 'run_placelt_accum_iter8'
#     T.ckpt = './pretrained_models/imagenet1k_vit_base.pth'
#
#     T.dataset = 'Place'
#     T.nb_classes = 365
#
#     T.epochs = 30
#     T.device = '1,4,6,7'
#
#     T.batch = 64
#     T.accum_iter = 8
#
#     T.model = 'vit_base_patch16'
#     T.input_size = 224
#     T.drop_path = 0.1
#
#     T.clip_grad = None
#     T.weight_decay = 0.05
#     T.adamW2 = 0.99
#
#     T.lr = 0.008
#     T.blr = 1e-3
#     T.layer_decay = 0.75
#     T.min_lr = 1e-6
#     T.warmup_epochs = 5
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
#     T.loss = 'Bal_BCE'
#     T.bal_tau = 1.05
#     T.smoothing = 0.1
#
#     T.global_pool = True
#
#     T.seed = 0
#     T.prit = 1
#
#     T.num_workers = 16
#     T.master_port = 29503
#
#     T.finetune()
#
# run_placelt_accum_iter8()

