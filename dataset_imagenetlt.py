import json
import random
import sys
from torchvision import datasets, transforms


from tqdm import tqdm
import torch
import os
import numpy as np
from torchvision import datasets, transforms
import argparse
# from classes import CLASSES_INAT, CLASSES_Places, CLASSES, CUSTOM_TEMPLATES
import shutil
from models_extend.dinov2.models import build_model_dinov2


class DatasetLT(datasets.ImageFolder):

    def get_cls_num(self):
        cls_num = [0] * len(self.classes)
        for img in self.imgs:
            cls_num[img[1]] += 1
        return cls_num

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """

        try:
            path, target = self.samples[index]
            sample = self.loader(path)
            if self.transform is not None:
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)
        except:
            return self.__getitem__(index+1)

        return sample, target


class MaybeToTensor(transforms.ToTensor):
    """
    Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor, or keep as is if already a tensor.
    """

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image, numpy.ndarray or torch.tensor): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        if isinstance(pic, torch.Tensor):
            return pic
        return super().__call__(pic)


IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]

def make_normalize_transform(
        mean: IMAGENET_DEFAULT_MEAN,
        std: IMAGENET_DEFAULT_STD,
) -> transforms.Normalize:
    return transforms.Normalize(mean=mean, std=std)

def make_classification_eval_transform(
        *,
        resize_size: int = 256,
        interpolation=transforms.InterpolationMode.BICUBIC,
        crop_size: int = 224,
        mean: IMAGENET_DEFAULT_MEAN=IMAGENET_DEFAULT_MEAN,
        std: IMAGENET_DEFAULT_STD=IMAGENET_DEFAULT_STD,
) -> transforms.Compose:
    transforms_list = [
        transforms.Resize(resize_size, interpolation=interpolation),
        transforms.CenterCrop(crop_size),
        MaybeToTensor(),
        make_normalize_transform(mean=mean, std=std),
    ]
    return transforms.Compose(transforms_list)


def extract_features(data_path, save_root, model_e):


    device = 'cuda'
    # Use timm's names


    train_dataset = DatasetLT(data_path, transform=make_classification_eval_transform())
    cls_to_index = train_dataset.class_to_idx
    index_to_cls = {}
    for k, v in cls_to_index.items():
        index_to_cls[str(v)] = k

    numclasses = len(train_dataset.class_to_idx)
    # start_label, numclasses = train_dataset.re_arrange(split_idx=split_idx)

    sampler_train = torch.utils.data.SequentialSampler(train_dataset)

    data_loader = torch.utils.data.DataLoader(
        train_dataset, sampler=sampler_train,
        batch_size=128,
        num_workers=8,
        pin_memory=True,
        drop_last=False
    )

    model = model_e
    model.to(device)
    model.eval()


    if not os.path.exists(save_root):
        os.mkdir(save_root)

    current_label = 0
    current_feature = None


    feature_list = []

    batch_index = 0

    for images, labels in tqdm(data_loader):
        batch_index += 1
        print(batch_index)
        with torch.no_grad():
            labels = labels.type(torch.long).view(-1)

            unique_labels = torch.unique(labels)



            images = images.to(device)

            _, features = model(images)



            for one_label in unique_labels:
                if one_label == current_label:
                    if current_feature is None:
                        print(one_label)
                        print(labels)
                        current_feature = features[labels==one_label]
                    else:
                        current_feature = torch.cat((current_feature, features[labels==one_label]), dim=0)

                        if batch_index == len(data_loader) and current_label == numclasses-1:
                            current_feature = torch.mean(current_feature, dim=0, keepdim=True).cpu().numpy()
                            feature_list.append(current_feature)
                            save_path = os.path.join(save_root, index_to_cls[str(current_label)] + '.npy')
                            np.save(save_path, current_feature)

                else:
                    save_path = os.path.join(save_root, index_to_cls[str(current_label)]+'.npy')
                    current_feature = torch.mean(current_feature, dim=0, keepdim=True).cpu().numpy()
                    feature_list.append(current_feature)
                    np.save(save_path, current_feature)

                    current_label += 1
                    current_feature = features[labels==one_label]

                    if current_label == numclasses-1 and batch_index == len(data_loader):
                        current_feature = torch.mean(current_feature, dim=0, keepdim=True).cpu().numpy()
                        feature_list.append(current_feature)
                        save_path = os.path.join(save_root, index_to_cls[str(current_label)]+'.npy')
                        np.save(save_path, current_feature)
                        break

    all_feature = np.concatenate(feature_list, axis=0)
    print(all_feature.shape)
    save_path = os.path.join(save_root,  'all.npy')
    np.save(save_path, all_feature)

def delete_empty(root_path):
    count = 0
    folder_names = os.listdir(root_path)
    for folder_name in folder_names:
        file_names = os.listdir(os.path.join(root_path, folder_name))
        if len(file_names) == 0:
            command = "rm -rf " + os.path.join(root_path, "'" + folder_name+"'")
            print(command)
            os.system(command)

def delete_error_image(raw_dataset_path):
    class DatasetLT_clean_up(datasets.ImageFolder):

        def get_cls_num(self):
            cls_num = [0] * len(self.classes)
            for img in self.imgs:
                cls_num[img[1]] += 1
            return cls_num

        def __getitem__(self, index: int):
            """
            Args:
                index (int): Index

            Returns:
                tuple: (sample, target) where target is class_index of the target class.
            """
            path, target = self.samples[index]
            try:
                sample = self.loader(path)
            except:
                command = "rm -rf " + path
                os.system(command)
                print(path)
                print(1)
                return torch.zeros([3, 224, 224]), target

            if self.transform is not None:
                sample = self.transform(sample)

            return sample, target

    train_dataset = DatasetLT_clean_up(raw_dataset_path, transform=make_classification_eval_transform())

    sampler_train = torch.utils.data.SequentialSampler(train_dataset)

    data_loader = torch.utils.data.DataLoader(
        train_dataset, sampler=sampler_train,
        batch_size=128,
        num_workers=8,
        pin_memory=True,
        drop_last=False
    )

    for images, labels in tqdm(data_loader):
        pass
if __name__ == '__main__':
    # aug_dataset_path = "/home/szzhao/LT_project/vit_LT/data/ImageNet-LT/val"
    # save_root = "/home/szzhao/LT_project/vit_LT/granularity_data/imagenet_test_2"
    #
    # # ckpt_or = '/home/szzhao/LT_project/vit_LT/ckpt_1/run_dinov2_imagenetlt_CE_stage_1/ImageNet-LT/vit_base_patch14_dinov2/run_dinov2_imagenetlt_CE_stage_1/checkpoint.pth'
    # ckpt_good = "/home/szzhao/LT_project/vit_LT/ckpt_0/run_dinov2_imagenetlt/ImageNet-LT/vit_base_patch14_dinov2/run_dinov2_imagenetlt/checkpoint.pth"
    #
    # ckpt_input = ckpt_good
    #
    # model_e = build_model_dinov2(num_classes=1000,
    #                            class_to_idx={}, ckpt=ckpt_input)
    #
    # extract_features(aug_dataset_path, save_root, model_e)


    aug_dataset_path = "/home/szzhao/LT_project/vit_LT/data/iNat18/val"
    save_root = "/home/szzhao/LT_project/vit_LT/granularity_data/inat18_test_2"

    # ckpt_or = '/home/szzhao/LT_project/vit_LT/ckpt_1/run_dinov2_imagenetlt_CE_stage_1/ImageNet-LT/vit_base_patch14_dinov2/run_dinov2_imagenetlt_CE_stage_1/checkpoint.pth'
    ckpt_good = "/home/szzhao/LT_project/vit_LT/ckpt_0/run_dinov2_inat224_CE_loss/iNat18/vit_base_patch14_dinov2/run_dinov2_inat224_CE_loss/checkpoint.pth"

    ckpt_input = ckpt_good

    model_e = build_model_dinov2(num_classes=8142,
                               class_to_idx={}, ckpt=ckpt_input)

    extract_features(aug_dataset_path, save_root, model_e)
