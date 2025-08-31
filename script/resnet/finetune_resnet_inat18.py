import os
import sys
sys.path.append("..")
sys.path.append(os.getcwd())
from util.trainer import Trainer
from util.trainer import EXP_PATH, WORK_PATH

def run_dinov2_inat18():
    T = Trainer()
    T.task = 'run_resnet_inat18_val'
    T.note = 'run_resnet_inat18_vak'
    T.ckpt = '/home/szzhao/pretrained_models/dinov2_vitb14_pretrain.pth'

    T.dataset = 'iNat18'
    T.nb_classes = 8142

    T.epochs = 400
    T.device = '0,1,2,3,4,5,6,7'

    T.batch = 128
    T.accum_iter = 1

    T.model = 'resnet50'
    T.input_size = 224
    T.drop_path = 0.1

    T.clip_grad = None
    T.weight_decay = 2e-4
    T.adamW2 = 0.99

    T.lr = 0.008
    T.blr = 2e-2
    T.layer_decay = 0.75
    T.min_lr = 0.
    T.warmup_epochs = 10
    T.clswarm = 0

    T.color_jitter = None
    T.aa = 'rand-m9-mstd0.5-inc1'

    T.reprob = 0.25
    T.remode = 'pixel'
    T.recount = 1
    T.resplit = False

    # T.mixup = 0.8
    # T.cutmix = 1.0
    # T.cutmix_minmax = None
    # T.mixup_prob = 1.0
    # T.mixup_switch_prob = 0.5
    # T.mixup_mode = 'batch'

    T.mixup = 0.01
    T.cutmix = 0.01
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
    T.gran_path = "/mnt/sda/szzhao/data/iNat18"

    T.num_workers = 16
    T.master_port = 29531

    T.mask_type = "unmask"
    # T.detach_aug = "detach_aug"

    T.finetune()

run_dinov2_inat18()
