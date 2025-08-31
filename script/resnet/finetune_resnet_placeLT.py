import os
import sys
sys.path.append("..")
sys.path.append(os.getcwd())
from util.trainer import Trainer
from util.trainer import EXP_PATH, WORK_PATH

def run_dinov2_placelt():
    T = Trainer()
    T.task = 'run_resnet_placelt_9'
    T.note = 'run_resnet_placelt_9'
    T.ckpt = '/home/szzhao/pretrained_models/dinov2_vitb14_pretrain.pth'

    T.dataset = 'Place'
    T.nb_classes = 365

    T.epochs = 50
    T.device = '0,1,2,3'

    T.batch = 128
    T.accum_iter = 1

    T.model = 'resnet152'
    T.input_size = 224
    T.drop_path = 0.1

    T.clip_grad = None
    T.weight_decay = 5e-4
    T.adamW2 = 0.99

    T.lr = 0.008
    T.blr = 2e-2
    T.layer_decay = 0.75
    T.min_lr = 0.
    T.warmup_epochs = 0

    T.color_jitter = None
    T.aa = 'rand-m9-mstd0.5-inc1'

    T.reprob = 0.25
    T.remode = 'pixel'
    T.recount = 1
    T.resplit = False
    T.clswarm = 0


    T.mixup = 0.001
    T.cutmix = 0.001
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
    T.master_port = 29507
    # T.clswarm = 3

    T.mask_type = "unmask"
    T.lp = 0

    T.gran_path = "/home/szzhao/LT_project/vit_LT/data/Place"

    T.finetune()

run_dinov2_placelt()
