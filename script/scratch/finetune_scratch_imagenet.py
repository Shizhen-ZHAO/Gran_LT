import os
import sys
sys.path.append("..")
sys.path.append(os.getcwd())
from util.trainer import Trainer
from util.trainer import EXP_PATH, WORK_PATH

def run_imagenetlt():
    T = Trainer()
    T.task = 'run_scratch_imagenet_val_1'
    T.note = 'run_scratch_imagenet_val_1'
    T.ckpt = './pretrained_models/ImageNet_checkpoint.pth'

    T.dataset = 'ImageNet-LT'
    T.nb_classes = 1000

    T.epochs = 100
    T.device = '2,3,4,5'

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
    T.min_lr = 0.0
    T.warmup_epochs = 10

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

    T.num_workers = 16
    T.master_port = 29502

    T.clswarm = 0
    T.lp = 0

    T.num_workers = 16
    T.master_port = 29506
    # T.gran_path = "./data/ImageNet-LT"
    T.gran_path = "/mnt/sdb/zsz/data/ImageNet-LT"
    T.aug_data = "no"
    T.clswarm_scale = 1

    T.mask_type = "unmask"

    T.finetune()

run_imagenetlt()

