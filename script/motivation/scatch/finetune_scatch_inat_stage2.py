import os
import sys
sys.path.append("..")
sys.path.append(os.getcwd())
from util.trainer import Trainer
from util.trainer import EXP_PATH, WORK_PATH

def run_dinov2_inat224():
    T = Trainer()
    T.task = 'run_scatch_inat224_CE_stage2'
    T.note = 'run_scatch_inat224_CE_stage2'
    T.ckpt = './ckpt/run_scatch_inat224_CE/iNat18/vit_base_patch14_dinov2/run_scatch_inat224_CE/checkpoint.pth'

    T.dataset = 'iNat18'
    T.nb_classes = 8142

    T.epochs = 10
    T.device = '0,1,2,3'

    T.batch = 128
    T.accum_iter = 4

    T.model = 'vit_base_patch14_dinov2'
    T.input_size = 224
    T.drop_path = 0.1

    T.clip_grad = None
    T.weight_decay = 0.05
    T.adamW2 = 0.99

    T.lr = 0.008
    T.blr = 1e-5
    T.layer_decay = 0.65
    T.min_lr = 0
    T.warmup_epochs = 0

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
    T.master_port = 29501

    T.clswarm = 0
    T.lp = 1

    T.finetune()

run_dinov2_inat224()

