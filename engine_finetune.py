# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
import copy
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import sys
from typing import Iterable, Optional
import torch.nn as nn

import torch
import h5py, os
import numpy as np
import torch.nn.functional as F

from timm.data import Mixup
from timm.utils import accuracy

import util.misc as misc
import util.lr_sched as lr_sched

from tqdm import tqdm

from util.loss import *

# criterion_SupCon = SupConLoss()

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None, tokens=None, resnet=False):
    model.train(True)

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    torch.cuda.empty_cache()

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # labels = targets


        # tokens_stack = torch.cat([tokens[targets[i]] for i in range(len(targets))], dim=0)
        digital_targets = targets
        if mixup_fn is not None:

            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():

            if resnet:
                logits = model(samples)
            else:
                logits, text_features = model(samples, digital_targets=digital_targets, detach_aug=args.detach_aug, normal_mask=args.normal_mask)
                # print(digital_targets.shape)
                # suploss = criterion_SupCon(text_features, labels=digital_targets)

            if args.loss == 'CE':
                loss = criterion(logits, targets)
            else:
                loss = criterion(logits, targets, digital_targets=digital_targets)

        # if text_features != []:
        #     print(text_features.shape)

        loss_value = loss.item()
        # loss_value_supcon = suploss.item()
        # loss_value = loss_value + loss_value_supcon

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])
        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, args):
    criterion = torch.nn.CrossEntropyLoss()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    model_eval = copy.deepcopy(model)
    class_to_idx_au = args.class_to_idx
    class_to_idx = args.class_to_idx_val

    # print(class_to_idx)
    # sys.exit()

    if len(class_to_idx_au) > len(class_to_idx):

        class_to_idx_2 = {}

        idx_to_idx = np.array([0] * len(class_to_idx))

        for cls, idx in class_to_idx.items():
            class_to_idx_2[cls] = class_to_idx_au[cls]
            idx_to_idx[idx] = class_to_idx_au[cls]

        # for name, p in model_eval.named_parameters():
        #     if "head" in name:
        #         print(name)
        #         # p = p[idx_to_idx]
        #         print(p.shape)
        #         # print(p)
        model_eval.module.head.weight = nn.Parameter(model_eval.module.head.weight[idx_to_idx])
        try:
            model_eval.module.head.bias = nn.Parameter(model_eval.module.head.bias[idx_to_idx])
        except:
            print('np bias')
        # model_eval.module.fc.weight = nn.Parameter(model_eval.module.fc.weight[idx_to_idx])
        # try:
        #     model_eval.module.fc.bias = nn.Parameter(model_eval.module.fc.bias[idx_to_idx])
        # except:
        #     print('np bias')

    # sys.exit()

    # switch to evaluation mode
    model_eval.eval()
    for batch in metric_logger.log_every(data_loader, 100, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():

            if 'resnet' in args.model:
                output = model_eval(images)
            else:
                output, _ = model_eval(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
        .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def get_shot(cls_num_list, args=None):
    # FIXME here follow TADE
    shot = {}
    cls_num_list = torch.tensor(cls_num_list)
    many_shot = cls_num_list > 100
    few_shot = cls_num_list < 20
    medium_shot = ~many_shot & ~few_shot

    shot['many_shot'] = many_shot
    shot['few_shot'] = few_shot
    shot['medium_shot'] = medium_shot

    return shot

# def get_shot(cls_num_list, args=None):
#     shot = {}
#     cls_num_list = torch.tensor(cls_num_list)
#     many_shot = cls_num_list > 100
#     few_shot = cls_num_list < 20
#     medium_shot = ~many_shot & ~few_shot
#
#     # sys.exit()
#     # path = "/home/szzhao/LT_project/vit_LT/granularity_data/inat_18_v3"
#     shot['many_shot'] = np.load(os.path.join(args.gran_path, 'many_shot.npy'), allow_pickle=True)
#     shot['few_shot'] = np.load(os.path.join(args.gran_path, 'few_shot.npy'), allow_pickle=True)
#     shot['medium_shot'] = ~(shot['many_shot'] | shot['few_shot'])
#
#     return shot


def calibration(preds, labels, confidences, num_bins=15):
    assert(len(confidences) == len(preds))
    assert(len(confidences) == len(labels))
    assert(num_bins > 0)

    bin_size = 1.0 / num_bins
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    indices = np.digitize(confidences, bins, right=True)

    bin_accuracies = np.zeros(num_bins, dtype=float)
    bin_confidences = np.zeros(num_bins, dtype=float)
    bin_counts = np.zeros(num_bins, dtype=int)

    for b in range(num_bins):
        selected = np.where(indices == b + 1)[0]
        if len(selected) > 0:
            bin_accuracies[b] = np.mean(labels[selected] == preds[selected])
            bin_confidences[b] = np.mean(confidences[selected])
            bin_counts[b] = len(selected)

    avg_acc = np.sum(bin_accuracies * bin_counts) / np.sum(bin_counts)
    avg_conf = np.sum(bin_confidences * bin_counts) / np.sum(bin_counts)

    gaps = np.abs(bin_accuracies - bin_confidences)
    ece = np.sum(gaps * bin_counts) / np.sum(bin_counts)
    mce = np.max(gaps)

    return { "accuracies": bin_accuracies, 
             "confidences": bin_confidences, 
             "counts": bin_counts, 
             "bins": bins,
             "avg_accuracy": avg_acc,
             "avg_confidence": avg_conf,
             "expected_calibration_error": ece,
             "max_calibration_error": mce }


@torch.no_grad()
def evaluate_all_metric(data_loader, model, device, args):
    model.eval()
    nClasses = len(args.cls_num)

    shot = get_shot(args.cls_num, args=args)

    many_shot = shot['many_shot']
    medium_shot = shot['medium_shot']
    few_shot = shot['few_shot']

    predList = np.array([])
    cfdsList = np.array([])
    grndList = np.array([])
    for images, labels in tqdm(data_loader):
        with torch.no_grad():
            images = images.to(device)
            labels = labels.type(torch.long).view(-1).numpy()
            if 'resnet' in args.model:
                logits = model(images)
            else:
                logits,_ = model(images)
            cfds, preds = F.softmax(logits.detach(), dim=1).max(dim=1)
            cfds = cfds.detach().squeeze().cpu().numpy()
            preds = preds.detach().squeeze().cpu().numpy()
            cfdsList = np.concatenate((cfdsList, cfds))
            predList = np.concatenate((predList, preds))
            grndList = np.concatenate((grndList, labels))

    results_dict = {
        "cfdsList": cfdsList,
        "predList": predList,
        "grndList": grndList
    }

    np.save('./prediction_results/scratch_imagenet_data.npy', results_dict)
    #
    cali = calibration(predList, grndList, cfdsList, num_bins=15)
    ece = cali['expected_calibration_error']
    mce = cali['max_calibration_error']
    
    cfd_per_class = [0] * nClasses
    pdt_per_class = [0] * nClasses
    rgt_per_class = [0] * nClasses
    acc_per_class = [0] * nClasses
    gts_per_class = [0] * nClasses
    
    cfd_map = [[0] * nClasses for _ in range(nClasses)]
    cfd_cnt = [[0] * nClasses for _ in range(nClasses)]
    
    for c, g, p in zip(cfdsList, grndList, predList):
        cfd_map[int(p)][int(g)] += c
        cfd_cnt[int(p)][int(g)] += 1
        gts_per_class[int(g)] += 1
        pdt_per_class[int(p)] += 1
        if g == p:
            cfd_per_class[int(g)] += c
            rgt_per_class[int(g)] += 1
            
    for i in range(nClasses):
        cnt = rgt_per_class[i]
        if cnt != 0:
            acc_per_class[i] = np.round(cnt/gts_per_class[i] * 100, decimals=2)
            cfd_per_class[i] = np.round(cfd_per_class[i]/cnt * 100, decimals=2)
    
    for i in range(nClasses):
        for j in range(nClasses):
            if cfd_cnt[i][j] != 0:
                cfd_map[i][j] = cfd_map[i][j] / cfd_cnt[i][j]


    avg_acc = np.sum(rgt_per_class) / np.sum(gts_per_class)
    acc_per_class = np.array(acc_per_class)
    many_shot_acc = acc_per_class[many_shot].mean()
    medium_shot_acc = acc_per_class[medium_shot].mean()
    few_shot_acc = acc_per_class[few_shot].mean()

    pdt_per_class = np.array(pdt_per_class)
    gts_per_class = np.array(gts_per_class)
    cls_num = np.array(args.cls_num)
    q = pdt_per_class / np.sum(pdt_per_class)
    pt = gts_per_class / np.sum(gts_per_class)
    ps = cls_num / np.sum(cls_num)

    pdc_s = np.sum(pt * np.log(pt + 1e-6) - pt * np.log(ps + 1e-6))
    pdc_t = np.sum(pt * np.log(pt + 1e-6) - pt * np.log(q + 1e-6))


    result = {
        'avg_acc': np.round(avg_acc*100, decimals=2).tolist(),
        'ece': np.round(ece*100, decimals=2).tolist(),
        'mce': np.round(mce*100, decimals=2).tolist(),
        'many' : np.round(many_shot_acc, decimals=2).tolist(),
        'medium' : np.round(medium_shot_acc, decimals=2).tolist(),
        'few' : np.round(few_shot_acc, decimals=2).tolist(),
        'pdc': np.round(float(pdc_t / pdc_s), decimals=2)
    }

    print(result)

    return result, np.array(cfd_cnt)

