import os
import sys
sys.path.append("..")
sys.path.append(os.getcwd())
from util.trainer import Trainer
from util.trainer import EXP_PATH, WORK_PATH

def run_clip_imagenetlt():
    T = Trainer()
    T.task = 'run_clip_imagenetlt_4'
    T.note = 'run_clip_imagenetlt_4'
    T.ckpt = ''

    T.dataset = 'ImageNet-LT'
    T.nb_classes = 1000

    T.epochs = 100
    T.device = '0,1,2,3'

    T.batch = 128
    T.accum_iter = 1

    T.model = 'vit_base_patch16_clip'
    T.input_size = 224
    T.drop_path = 0.1

    T.clip_grad = None
    T.weight_decay = 0.05
    T.adamW2 = 0.99

    T.lr = 0.008
    T.blr = 2.5e-6
    T.layer_decay = 0
    T.min_lr = 0.
    T.warmup_epochs = 10

    T.color_jitter = None
    T.aa = 'rand-m9-mstd0.5-inc1'

    T.reprob = 0.25
    T.remode = 'pixel'
    T.recount = 1
    T.resplit = False

    T.mixup = 0.001
    T.cutmix = 0.001
    T.cutmix_minmax = None
    T.mixup_prob = 1.0
    T.mixup_switch_prob = 0.5
    T.mixup_mode = 'batch'

    # T.loss = 'CB_CE'
    T.loss = 'Bal_CE'
    T.bal_tau = 1.0
    T.smoothing = 0.1

    T.global_pool = True

    T.text_classifier=True

    T.seed = 0
    T.prit = 1
    T.lp = 0

    T.clswarm = 3

    T.num_workers = 16
    T.master_port = 29507
    T.gran_path = "/home/szzhao/LT_project/vit_LT/data/ImageNet-LT"
    T.aug_data = "no"

    T.mask_type = "unmask"

    T.finetune()

run_clip_imagenetlt()

