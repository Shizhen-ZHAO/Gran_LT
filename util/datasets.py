# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import sys

import PIL
from PIL import ImageFilter
import torch
import torchvision
import numpy as np
from torchvision import datasets, transforms
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from PIL import Image
from chatgpt_text_generation.imagenet.imagenet_classes import CLASSES_IMAGENET
from clip.simple_tokenizer import SimpleTokenizer
# from chatgpt_text_generation.imagenet.imagenet_text_postprocessing import CLASSNAME2TEXT
from tqdm import tqdm
import random
from .randaugment import rand_augment_transform

from multiprocessing import Pool
# class LT_Dataset(Dataset):
#
#     def __init__(self, root, txt, transform=None):
#         self.img_path = []
#         self.labels = []
#         self.transform = transform
#         with open(txt) as f:
#             for line in f:
#                 self.img_path.append(os.path.join(root, line.split()[0]))
#                 self.labels.append(int(line.split()[1]))
#
#     def __len__(self):
#         return len(self.labels)
#
#     # def get_cls_num(self):
#     #     cls_num = [0] * len(self.classes)
#     #     for img in self.imgs:
#     #         cls_num[img[1]] += 1
#     #     return cls_num
#
#     def __getitem__(self, index):
#
#         path = self.img_path[index]
#         label = self.labels[index]
#
#         with open(path, 'rb') as f:
#             sample = Image.open(f).convert('RGB')
#
#         if self.transform is not None:
#             sample = self.transform(sample)
#
#         return sample, label


class CIFAR10_LT(torchvision.datasets.CIFAR10):
    cls_num = 10

    def __init__(self,
                 root,
                 imb_type='exp',
                 imb_factor=0.01,
                 rand_number=0,
                 train=True,
                 transform=None,
                 target_transform=None,
                 download=False):
        super(CIFAR10_LT, self).__init__(root, train, transform,
                                               target_transform, download)
        np.random.seed(rand_number)
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        if train:
            img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type,imb_factor)
            self.gen_imbalanced_data(img_num_list)

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([
                the_class,
            ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets

    def get_cls_num(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list


class CIFAR100_LT(CIFAR10_LT):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
    cls_num = 100

# class DatasetLT(datasets.ImageFolder):
#
#     def __init__(self, root, transform=None):
#         super().__init__(root, transform=transform)
#         # self.imgs = self.samples
#
#         self.descriptions = CLASSNAME2TEXT
#         new_names = [""] * len(self.classes)
#         self.cls_names = CLASSES_IMAGENET
#         # for k, v in self.class_to_idx.items():
#         #     original_id = int(k)
#         #     new_id = int(v)
#         #     new_names[new_id] = self.cls_names[original_id]
#         self.cls_names = new_names
#
#         self._tokenizer = SimpleTokenizer()
#
#         self.descriptions_list = []
#         # for cls_name in self.cls_names:
#         #     self.descriptions_list.append(self.descriptions[cls_name])
#
#         self.token_list = []
#         # self.token_list = self.tokenize(self.descriptions_list)
#
#
#     def tokenize(self, texts, context_length=75):
#         """
#         modified from CLIP
#
#         Returns the tokenized representation of given input string(s)
#
#         Parameters
#         ----------
#         texts : Union[str, List[str]]
#             An input string or a list of input strings to tokenize
#
#         Returns
#         -------
#         A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
#         """
#
#         def _tokenize(texts):
#             sot_token = self._tokenizer.encoder["<|startoftext|>"]  # 49406
#             eot_token = self._tokenizer.encoder["<|endoftext|>"]  # 49407
#             all_tokens = [[sot_token] + self._tokenizer.encode(text)[:context_length] + [eot_token] for text in
#                           texts]
#             result = torch.zeros(len(all_tokens), context_length + 2, dtype=torch.long)
#             for i, tokens in enumerate(all_tokens):
#                 if len(tokens) > context_length + 2:
#                     raise RuntimeError(
#                         f"Input {texts[i]} is too long for context length {context_length}")
#                 result[i, :len(tokens)] = torch.tensor(tokens)
#             return result
#
#         print(texts[0])
#         if isinstance(texts, str):
#             texts = [texts]
#         elif isinstance(texts[0], list):
#             return [_tokenize(text) for text in tqdm(texts)]
#             # return [{_tokenize(v) for k, v in tqdm(enumerate(texts))]
#         return _tokenize(texts)
#
#     def get_cls_num(self):
#         cls_num = [0] * len(self.classes)
#         for img in self.imgs:
#             cls_num[img[1]] += 1
#         return cls_num
#
#     def __getitem__(self, index: int):
#         """
#         Args:
#             index (int): Index
#
#         Returns:
#             tuple: (sample, target) where target is class_index of the target class.
#         """
#         print(index)
#         path, target = self.samples[index]
#         sample = self.loader(path)
#         if self.transform is not None:
#             sample = self.transform(sample)
#         if self.target_transform is not None:
#             target = self.target_transform(target)
#
#         return sample, target

def build_dataset(is_train, args):
    transform = build_transform(is_train, args)
    if args.dataset == 'cifar10-LT':
        dataset = CIFAR10_LT(root=args.data_path,
                            imb_type='exp',
                            imb_factor=1/args.imbf,
                            rand_number=0,
                            train=is_train,
                            download=True,
                            transform=transform)
    elif args.dataset == 'cifar100-LT':
        dataset = CIFAR100_LT(root=args.data_path,
                            imb_type='exp',
                            imb_factor=1/args.imbf,
                            rand_number=0,
                            train=is_train,
                            download=True,
                            transform=transform)
    else:

        root = os.path.join(args.gran_path, 'train' if is_train else 'val')

        # if is_train:
        #     root = os.path.join(args.data_path, 'train')
        # else:
        #     root = os.path.join(args.data_path, 'train' if is_train else 'val')

        if args.dataset == 'iNat18' and args.aug_data == "yes":
            dataset = DatasetLT_aug(root, transform=transform, aug_data=args.aug_data)
        else:
            dataset = DatasetLT(root, transform=transform, aug_data=args.aug_data)
    # print(dataset)
    return dataset

def build_dataset_train(is_train, args):
    transform = build_transform(is_train, args)
    if args.dataset == 'ImageNet-LT':
        txt = "./data_tools/ImageNet_LT/ImageNet_LT_train.txt"
        root = "/mnt/sda/zsz/VL_LT/data/ImageNet"
    elif args.dataset == 'iNat18':
        txt = "./data_tools/iNaturalist18/iNaturalist18_train.txt"
        root = "/mnt/sda/zsz/VL_LT/data/iNat"
    elif args.dataset == 'Place':
        txt = "./data_tools/Places_LT/Places_LT_train.txt"
        root = "/mnt/sda/zsz/VL_LT/data/Places"
    dataset = LT_Dataset(root, txt, transform)
    return dataset

def build_dataset_place_test(is_train, args):
    transform = build_transform(is_train, args)
    root = os.path.join(args.data_path, 'test')
    dataset = DatasetLT(root, transform=transform)
    return dataset

def get_mean_std(args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    if args.dataset == 'ImageNet-LT':
        mean = [0.479672, 0.457713, 0.407721]
        std = [0.278976, 0.271203, 0.286062]
    elif args.dataset == 'iNat18':
        mean = [0.466, 0.471, 0.380]
        std = [0.195, 0.194, 0.192]
    elif args.dataset == 'ImageNet-BAL':
        mean = [0.480767, 0.457071, 0.407718]
        std = [0.279940, 0.272481, 0.286038]
    elif args.dataset == 'ImageNet':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif args.dataset == 'Place':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else: # Cifar 10 or 100
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
    return mean, std



# def build_transform(is_train, args):
#     # mean, std = get_mean_std(args)
#
#     mean = (0.4815, 0.4578, 0.4082)
#     std = (0.2686, 0.2613, 0.2758)
#     # train transform
#     if is_train:
#         transform = transforms.Compose([
#             transforms.RandomResizedCrop(224),
#             transforms.RandomHorizontalFlip(),
#             transforms.ColorJitter(
#                 brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
#             transforms.ToTensor(),
#             transforms.Normalize(mean, std)
#         ])
#
#         return transform
#
#     # eval transform
#
#     crop_pct = 224 / 256 if args.input_size <= 224 else 1.0
#     size = int(args.input_size / crop_pct)
#     # to maintain same ratio w.r.t. 224 images
#     # from torchvision import datasets, transforms
#
#     test_transform = transforms.Compose([
#         transforms.Resize(size, interpolation=PIL.Image.BICUBIC),
#         transforms.CenterCrop(args.input_size),
#         transforms.ToTensor(),
#         transforms.Normalize(mean, std)
#     ])
#     return test_transform


def build_transform_resnet(mean, std, args):

    class GaussianBlur(object):
        """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

        def __init__(self, sigma=[.1, 2.]):
            self.sigma = sigma

        def __call__(self, x):
            sigma = random.uniform(self.sigma[0], self.sigma[1])
            x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
            return x

    if 'resnet50' in args.model:

        rgb_mean = (0.485, 0.456, 0.406)
        ra_params = dict(translate_const=int(224 * 0.45), img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]),)
        augmentation = [
                transforms.RandomResizedCrop(224, scale=(0.08, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)
                ], p=1.0),
                rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(2, 10), ra_params),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
        ]
    else:
        augmentation = [
                transforms.RandomResizedCrop(224),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)  # not strengthened
                ], p=1.0),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
        ]

    return transforms.Compose(augmentation)

def build_transform(is_train, args):

    mean, std = get_mean_std(args)

    # train transform
    if is_train:
        # train transform
        if 'clip' in args.model:
            mean = (0.4815, 0.4578, 0.4082)
            std = (0.2686, 0.2613, 0.2758)

            if args.dataset == 'ImageNet-LT':
                transform = transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(
                        brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)
                ])

            # this should always dispatch to transforms_imagenet_train
            else:
                transform = create_transform(
                input_size=args.input_size,
                is_training=True,
                color_jitter=args.color_jitter,
                auto_augment=args.aa,
                interpolation='bicubic',
                re_prob=args.reprob,
                re_mode=args.remode,
                re_count=args.recount,
                mean=mean,
                std=std)

        else:

            transform = create_transform(
                input_size=args.input_size,
                is_training=True,
                color_jitter=args.color_jitter,
                auto_augment=args.aa,
                interpolation='bicubic',
                re_prob=args.reprob,
                re_mode=args.remode,
                re_count=args.recount,
                mean=mean,
                std=std)

            if 'resnet' in args.model:
                transform = build_transform_resnet(mean, std, args)

        return transform


    # eval transform

    crop_pct = 224 / 256 if args.input_size <= 224 else 1.0
    size = int(args.input_size / crop_pct)
    # to maintain same ratio w.r.t. 224 images
    # from torchvision import datasets, transforms

    test_transform = transforms.Compose([
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),
        transforms.CenterCrop(args.input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    return test_transform



class DatasetLT(datasets.ImageFolder):

    def __init__(
            self,
            root = None,
            transform = None,
            aug_data = 'no'
    ):
        super().__init__(
            root = root,
            transform=transform,
        )

        # self.descriptions = CLASSNAME2TEXT
        new_names = [""] * len(self.classes)
        self.cls_names = CLASSES_IMAGENET
        # for k, v in self.class_to_idx.items():
        #     original_id = int(k)
        #     new_id = int(v)
        #     new_names[new_id] = self.cls_names[original_id]
        self.cls_names = new_names

        self._tokenizer = SimpleTokenizer()

        self.descriptions_list = []
        # for cls_name in self.cls_names:
        #     self.descriptions_list.append(self.descriptions[cls_name])

        self.token_list = []
        # self.token_list = self.tokenize(self.descriptions_list)

        self.all_samples = [[] for i in range(len(self.classes))]
        for index in range(len(self.samples)):
            path, target = self.samples[index]
            item = path, target
            self.all_samples[target].append(item)

        self.idx_to_class = {}
        for clsname, idx in self.class_to_idx.items():
            self.idx_to_class[str(idx)] = clsname
        self.cls_num = self.get_cls_num()


        self.root = root

        self.conp_cls = self.root[:-len(self.root.split('/')[-1])] + "conp_cls.npy"

        self.added_clsname = []
        if os.path.exists(self.conp_cls):
            conp_cls = np.load(self.conp_cls, allow_pickle=True).item()
            for imagenet_cls, cls_list in conp_cls.items():
                self.added_clsname += cls_list

        self.aug_data = aug_data
        if aug_data == 'yes':
            self.re_arrange()
        self.imgs = self.samples


    def get_cls_num(self):
        cls_num = [0] * len(self.classes)
        for img in self.imgs:
            cls_num[img[1]] += 1
        return cls_num

    def re_arrange(self):
        self.new_samples = []
        wordname_to_num_path =  self.root[:-len(self.root.split('/')[-1])] + "wordname_to_num.npy"
        self.wordname_to_num = np.load(wordname_to_num_path, allow_pickle=True).item()

        for cls_index in range(len(self.classes)):
            cls_name = self.idx_to_class[str(cls_index)]
            if cls_name in self.added_clsname:
                # random.shuffle(self.all_samples[cls_index])
                # anchor_cls = cls_name.split('_a_')[0]
                # anchor_idx = self.class_to_idx[anchor_cls]
                cls_num = self.wordname_to_num[cls_name]
                random.shuffle(self.all_samples[cls_index])

                if cls_num > 100:
                    selected_num = 50
                elif cls_num > 20:
                    selected_num = cls_num
                else:
                    selected_num = 50
                self.new_samples = self.new_samples + self.all_samples[cls_index][:selected_num]
            else:
                self.new_samples = self.new_samples + self.all_samples[cls_index]

        self.samples = self.new_samples

    def generate_mask(self, all_normal=True, mask_type=""):
        # return None, None
        class_to_idx = self.class_to_idx
        cls_num = len(class_to_idx)

        if self.aug_data == "no":

            mask = np.ones((cls_num, cls_num))
            normal_mask = np.ones(cls_num)

            return mask, normal_mask

        conp_cls_path = self.root[:-len(self.root.split('/')[-1])] + "conp_cls.npy"

        conp_cls = np.load(conp_cls_path, allow_pickle=True).item()

        added2normal = {}
        for imagenet_cls, cls_list in conp_cls.items():

            for added_name in cls_list:
                if added_name not in added2normal:
                    added2normal[added_name] = []
                if imagenet_cls not in added2normal[added_name]:
                    added2normal[added_name].append(imagenet_cls)


        normal_mask = np.zeros(cls_num)

        for cls_name, idx in class_to_idx.items():
            if 'n' not in cls_name:
                normal_mask[idx] = 1

        normal_idx = np.where(normal_mask == 1)[0]
        added_idx = np.where(normal_mask == 0)[0]
        cls_num_list = self.get_cls_num()

        if mask_type == "no_fm":

            mask = np.ones((cls_num, cls_num))

            for imagenet_cls, cls_list in conp_cls.items():

                cls_index = class_to_idx[imagenet_cls]
                if cls_num_list[cls_index] < 100:

                    for add_cls in cls_list:
                        if add_cls in class_to_idx:
                            add_index = class_to_idx[add_cls]
                            mask[cls_index, add_index] = 0

            for added_cls, normal_list in added2normal.items():
                if added_cls in class_to_idx:
                    added_index = class_to_idx[added_cls]

                    for normal_name in normal_list:
                        normal_index = class_to_idx[normal_name]
                        if cls_num_list[normal_index] < 100:
                            mask[added_index, normal_index] = 0

        elif mask_type == "no_fm_remove_normal_neg":

            mask = np.ones((cls_num, cls_num))

            for imagenet_cls, cls_list in conp_cls.items():

                cls_index = class_to_idx[imagenet_cls]
                # if cls_num_list[cls_index] < 100:

                for add_cls in cls_list:
                    if add_cls in class_to_idx:
                        add_index = class_to_idx[add_cls]
                        mask[cls_index, add_index] = 0

            for added_cls, normal_list in added2normal.items():
                if added_cls in class_to_idx:
                    added_index = class_to_idx[added_cls]

                    for normal_name in normal_list:
                        normal_index = class_to_idx[normal_name]
                        # if cls_num_list[normal_index] < 100:
                        mask[added_index, normal_index] = 0

        elif mask_type == "no_fm_remove_normal_neg_back":

            mask = np.ones((cls_num, cls_num))

            for imagenet_cls, cls_list in conp_cls.items():

                cls_index = class_to_idx[imagenet_cls]
                # if cls_num_list[cls_index] < 100:

                for add_cls in cls_list:
                    if add_cls in class_to_idx:
                        add_index = class_to_idx[add_cls]
                        mask[cls_index, add_index] = 0

            for added_cls, normal_list in added2normal.items():
                if added_cls in class_to_idx:
                    added_index = class_to_idx[added_cls]

                    mask[added_index, added_idx] = 0
                    mask[added_index, added_index] = 1

                    for normal_name in normal_list:
                        normal_index = class_to_idx[normal_name]
                        # if cls_num_list[normal_index] < 100:
                        mask[added_index, normal_index] = 0


        elif mask_type == "no_fm_1":




            mask = np.ones((cls_num, cls_num))

            for imagenet_cls, cls_list in conp_cls.items():

                cls_index = class_to_idx[imagenet_cls]
                if cls_num_list[cls_index] < 100:

                    for add_cls in cls_list:
                        if add_cls in class_to_idx:
                            add_index = class_to_idx[add_cls]
                            mask[cls_index, add_index] = 0

        elif mask_type == "no_fm_2":
            mask = np.ones((cls_num, cls_num))

            for added_cls, normal_list in added2normal.items():
                if added_cls in class_to_idx:
                    added_index = class_to_idx[added_cls]

                    for normal_name in normal_list:
                        normal_index = class_to_idx[normal_name]
                        if cls_num_list[normal_index] < 100:
                            mask[added_index, normal_index] = 0


        elif mask_type == "no_fm_2_back":
            mask = np.ones((cls_num, cls_num))

            for added_cls, normal_list in added2normal.items():
                if added_cls in class_to_idx:
                    added_index = class_to_idx[added_cls]

                    mask[added_index, added_idx] = 0
                    mask[added_index, added_index] = 1

                    for normal_name in normal_list:
                        normal_index = class_to_idx[normal_name]
                        if cls_num_list[normal_index] < 100:
                            mask[added_index, normal_index] = 0

        elif mask_type == "no_fm_2_back_remove_normal":
            mask = np.ones((cls_num, cls_num))

            for added_cls, normal_list in added2normal.items():
                if added_cls in class_to_idx:
                    added_index = class_to_idx[added_cls]

                    mask[added_index, added_idx] = 0
                    mask[added_index, added_index] = 1

                    mask[added_index, normal_idx] = 0

                    # for normal_name in normal_list:
                    #     normal_index = class_to_idx[normal_name]
                    #     if cls_num_list[normal_index] < 100:
                    #         mask[added_index, normal_index] = 0

        elif mask_type == "no_fm_2_remove_normal_neg_back":
            # print("no_fm_2_remove_normal_neg")
            # sys.exit()
            mask = np.ones((cls_num, cls_num))

            for added_cls, normal_list in added2normal.items():
                if added_cls in class_to_idx:
                    added_index = class_to_idx[added_cls]

                    mask[added_index, added_idx] = 0
                    mask[added_index, added_index] = 1

                    for normal_name in normal_list:
                        normal_index = class_to_idx[normal_name]
                        # if cls_num_list[normal_index] < 100:
                        mask[added_index, normal_index] = 0

        elif mask_type == "no_fm_2_remove_normal_neg":
            # print("no_fm_2_remove_normal_neg")
            # sys.exit()
            mask = np.ones((cls_num, cls_num))

            for added_cls, normal_list in added2normal.items():
                if added_cls in class_to_idx:
                    added_index = class_to_idx[added_cls]

                    for normal_name in normal_list:
                        normal_index = class_to_idx[normal_name]
                        # if cls_num_list[normal_index] < 100:
                        mask[added_index, normal_index] = 0
        elif mask_type == "normal":

            cls_num = len(class_to_idx)

            mask = np.zeros((cls_num, cls_num))

            normal_mask = np.zeros(cls_num)

            for cls_name, idx in class_to_idx.items():
                if 'n' not in cls_name:
                    normal_mask[idx] = 1

            normal_idx = np.where(normal_mask == 1)[0]
            added_idx = np.where(normal_mask == 0)[0]

            for imagenet_cls, cls_list in conp_cls.items():

                cls_index = class_to_idx[imagenet_cls]
                mask[cls_index, normal_idx] = 1

                for add_cls in cls_list:
                    if add_cls in class_to_idx:
                        add_index = class_to_idx[add_cls]
                        mask[cls_index, add_index] = 1

            for added_cls, normal_list in added2normal.items():
                if added_cls in class_to_idx:
                    added_index = class_to_idx[added_cls]

                    if all_normal:
                        mask[added_index, normal_idx] = 1
                    else:
                        for normal_name in normal_list:
                            normal_index = class_to_idx[normal_name]
                            mask[added_index, normal_index] = 1
                    mask[added_index, added_index] = 1
        else:
            mask = np.zeros((cls_num, cls_num))
            normal_mask = np.zeros(cls_num)

        return mask, normal_mask


    def __getitem__(self, index: int):
        try:
            path, target = self.samples[index]
            sample = self.loader(path)
            if self.transform is not None:
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)
        except:
            return self.__getitem__(index + 1)
        return sample, target


# class DatasetLT(datasets.ImageFolder):
#
#     def __init__(
#             self,
#             root = None,
#             transform = None,
#             aug_data = 'no'
#     ):
#         super().__init__(
#             root = root,
#             transform=transform,
#         )
#
#
#         # self.descriptions = CLASSNAME2TEXT
#         new_names = [""] * len(self.classes)
#         self.cls_names = CLASSES_IMAGENET
#         # for k, v in self.class_to_idx.items():
#         #     original_id = int(k)
#         #     new_id = int(v)
#         #     new_names[new_id] = self.cls_names[original_id]
#         self.cls_names = new_names
#
#         self._tokenizer = SimpleTokenizer()
#
#         self.descriptions_list = []
#         # for cls_name in self.cls_names:
#         #     self.descriptions_list.append(self.descriptions[cls_name])
#
#         self.token_list = []
#         # self.token_list = self.tokenize(self.descriptions_list)
#
#         self.all_samples = [[] for i in range(len(self.classes))]
#         for index in range(len(self.samples)):
#             path, target = self.samples[index]
#             item = path, target
#             self.all_samples[target].append(item)
#
#         self.idx_to_class = {}
#         for clsname, idx in self.class_to_idx.items():
#             self.idx_to_class[str(idx)] = clsname
#         self.cls_num = self.get_cls_num()
#
#
#         self.root = root
#
#         self.conp_cls = self.root[:-len(self.root.split('/')[-1])] + "conp_cls.npy"
#
#         self.added_clsname = []
#         if os.path.exists(self.conp_cls):
#             conp_cls = np.load(self.conp_cls, allow_pickle=True).item()
#             for imagenet_cls, cls_list in conp_cls.items():
#                 self.added_clsname += cls_list
#
#         self.aug_data = aug_data
#         if aug_data == 'yes':
#             self.re_arrange()
#         self.imgs = self.samples
#
#
#     def get_cls_num(self):
#         cls_num = [0] * len(self.classes)
#         for img in self.imgs:
#             cls_num[img[1]] += 1
#         return cls_num
#
#     def re_arrange(self):
#         self.new_samples = []
#         wordname_to_num_path =  self.root[:-len(self.root.split('/')[-1])] + "wordname_to_num.npy"
#         self.wordname_to_num = np.load(wordname_to_num_path, allow_pickle=True).item()
#
#         for cls_index in range(len(self.classes)):
#             cls_name = self.idx_to_class[str(cls_index)]
#             if cls_name in self.added_clsname:
#                 # random.shuffle(self.all_samples[cls_index])
#                 # anchor_cls = cls_name.split('_a_')[0]
#                 # anchor_idx = self.class_to_idx[anchor_cls]
#                 cls_num = self.wordname_to_num[cls_name]
#                 random.shuffle(self.all_samples[cls_index])
#
#                 if cls_num > 100:
#                     selected_num = 50
#                 elif cls_num > 20:
#                     selected_num = cls_num
#                 else:
#                     selected_num = 50
#                 self.new_samples = self.new_samples + self.all_samples[cls_index][:selected_num]
#             else:
#                 self.new_samples = self.new_samples + self.all_samples[cls_index]
#
#         self.samples = self.new_samples
#
#     def generate_mask(self, all_normal=True, mask_type=""):
#         # return None, None
#         class_to_idx = self.class_to_idx
#         cls_num = len(class_to_idx)
#
#         if self.aug_data == "no":
#
#             mask = np.ones((cls_num, cls_num))
#             normal_mask = np.ones(cls_num)
#
#             return mask, normal_mask
#
#         conp_cls_path = self.root[:-len(self.root.split('/')[-1])] + "conp_cls.npy"
#
#         conp_cls = np.load(conp_cls_path, allow_pickle=True).item()
#
#         added2normal = {}
#         for imagenet_cls, cls_list in conp_cls.items():
#
#             for added_name in cls_list:
#                 if added_name not in added2normal:
#                     added2normal[added_name] = []
#                 if imagenet_cls not in added2normal[added_name]:
#                     added2normal[added_name].append(imagenet_cls)
#
#
#         normal_mask = np.zeros(cls_num)
#
#         for cls_name, idx in class_to_idx.items():
#             if 'n' not in cls_name:
#                 normal_mask[idx] = 1
#
#         normal_idx = np.where(normal_mask == 1)[0]
#         added_idx = np.where(normal_mask == 0)[0]
#         cls_num_list = self.get_cls_num()
#
#         if mask_type == "no_fm":
#
#             mask = np.ones((cls_num, cls_num))
#
#             for imagenet_cls, cls_list in conp_cls.items():
#
#                 cls_index = class_to_idx[imagenet_cls]
#                 if cls_num_list[cls_index] < 100:
#
#                     for add_cls in cls_list:
#                         if add_cls in class_to_idx:
#                             add_index = class_to_idx[add_cls]
#                             mask[cls_index, add_index] = 0
#
#             for added_cls, normal_list in added2normal.items():
#                 if added_cls in class_to_idx:
#                     added_index = class_to_idx[added_cls]
#
#                     for normal_name in normal_list:
#                         normal_index = class_to_idx[normal_name]
#                         if cls_num_list[normal_index] < 100:
#                             mask[added_index, normal_index] = 0
#
#         elif mask_type == "no_fm_remove_normal_neg":
#
#             mask = np.ones((cls_num, cls_num))
#
#             for imagenet_cls, cls_list in conp_cls.items():
#
#                 cls_index = class_to_idx[imagenet_cls]
#                 # if cls_num_list[cls_index] < 100:
#
#                 for add_cls in cls_list:
#                     if add_cls in class_to_idx:
#                         add_index = class_to_idx[add_cls]
#                         mask[cls_index, add_index] = 0
#
#             for added_cls, normal_list in added2normal.items():
#                 if added_cls in class_to_idx:
#                     added_index = class_to_idx[added_cls]
#
#                     for normal_name in normal_list:
#                         normal_index = class_to_idx[normal_name]
#                         # if cls_num_list[normal_index] < 100:
#                         mask[added_index, normal_index] = 0
#
#         elif mask_type == "no_fm_remove_normal_neg_back":
#
#             mask = np.ones((cls_num, cls_num))
#
#             for imagenet_cls, cls_list in conp_cls.items():
#
#                 cls_index = class_to_idx[imagenet_cls]
#                 # if cls_num_list[cls_index] < 100:
#
#                 for add_cls in cls_list:
#                     if add_cls in class_to_idx:
#                         add_index = class_to_idx[add_cls]
#                         mask[cls_index, add_index] = 0
#
#             for added_cls, normal_list in added2normal.items():
#                 if added_cls in class_to_idx:
#                     added_index = class_to_idx[added_cls]
#
#                     mask[added_index, added_idx] = 0
#                     mask[added_index, added_index] = 1
#
#                     for normal_name in normal_list:
#                         normal_index = class_to_idx[normal_name]
#                         # if cls_num_list[normal_index] < 100:
#                         mask[added_index, normal_index] = 0
#
#
#         elif mask_type == "no_fm_1":
#
#
#
#
#             mask = np.ones((cls_num, cls_num))
#
#             for imagenet_cls, cls_list in conp_cls.items():
#
#                 cls_index = class_to_idx[imagenet_cls]
#                 if cls_num_list[cls_index] < 100:
#
#                     for add_cls in cls_list:
#                         if add_cls in class_to_idx:
#                             add_index = class_to_idx[add_cls]
#                             mask[cls_index, add_index] = 0
#
#         elif mask_type == "no_fm_2":
#             mask = np.ones((cls_num, cls_num))
#
#             for added_cls, normal_list in added2normal.items():
#                 if added_cls in class_to_idx:
#                     added_index = class_to_idx[added_cls]
#
#                     for normal_name in normal_list:
#                         normal_index = class_to_idx[normal_name]
#                         if cls_num_list[normal_index] < 100:
#                             mask[added_index, normal_index] = 0
#
#
#         elif mask_type == "no_fm_2_back":
#             mask = np.ones((cls_num, cls_num))
#
#             for added_cls, normal_list in added2normal.items():
#                 if added_cls in class_to_idx:
#                     added_index = class_to_idx[added_cls]
#
#                     mask[added_index, added_idx] = 0
#                     mask[added_index, added_index] = 1
#
#                     for normal_name in normal_list:
#                         normal_index = class_to_idx[normal_name]
#                         if cls_num_list[normal_index] < 100:
#                             mask[added_index, normal_index] = 0
#
#         elif mask_type == "no_fm_2_back_remove_normal":
#             mask = np.ones((cls_num, cls_num))
#
#             for added_cls, normal_list in added2normal.items():
#                 if added_cls in class_to_idx:
#                     added_index = class_to_idx[added_cls]
#
#                     mask[added_index, added_idx] = 0
#                     mask[added_index, added_index] = 1
#
#                     mask[added_index, normal_idx] = 0
#
#                     # for normal_name in normal_list:
#                     #     normal_index = class_to_idx[normal_name]
#                     #     if cls_num_list[normal_index] < 100:
#                     #         mask[added_index, normal_index] = 0
#
#         elif mask_type == "no_fm_2_remove_normal_neg_back":
#             # print("no_fm_2_remove_normal_neg")
#             # sys.exit()
#             mask = np.ones((cls_num, cls_num))
#
#             for added_cls, normal_list in added2normal.items():
#                 if added_cls in class_to_idx:
#                     added_index = class_to_idx[added_cls]
#
#                     mask[added_index, added_idx] = 0
#                     mask[added_index, added_index] = 1
#
#                     for normal_name in normal_list:
#                         normal_index = class_to_idx[normal_name]
#                         # if cls_num_list[normal_index] < 100:
#                         mask[added_index, normal_index] = 0
#
#         elif mask_type == "no_fm_2_remove_normal_neg":
#             # print("no_fm_2_remove_normal_neg")
#             # sys.exit()
#             mask = np.ones((cls_num, cls_num))
#
#             for added_cls, normal_list in added2normal.items():
#                 if added_cls in class_to_idx:
#                     added_index = class_to_idx[added_cls]
#
#                     for normal_name in normal_list:
#                         normal_index = class_to_idx[normal_name]
#                         # if cls_num_list[normal_index] < 100:
#                         mask[added_index, normal_index] = 0
#         elif mask_type == "normal":
#
#             cls_num = len(class_to_idx)
#
#             mask = np.zeros((cls_num, cls_num))
#
#             normal_mask = np.zeros(cls_num)
#
#             for cls_name, idx in class_to_idx.items():
#                 if 'n' not in cls_name:
#                     normal_mask[idx] = 1
#
#             normal_idx = np.where(normal_mask == 1)[0]
#             added_idx = np.where(normal_mask == 0)[0]
#
#             for imagenet_cls, cls_list in conp_cls.items():
#
#                 cls_index = class_to_idx[imagenet_cls]
#                 mask[cls_index, normal_idx] = 1
#
#                 for add_cls in cls_list:
#                     if add_cls in class_to_idx:
#                         add_index = class_to_idx[add_cls]
#                         mask[cls_index, add_index] = 1
#
#             for added_cls, normal_list in added2normal.items():
#                 if added_cls in class_to_idx:
#                     added_index = class_to_idx[added_cls]
#
#                     if all_normal:
#                         mask[added_index, normal_idx] = 1
#                     else:
#                         for normal_name in normal_list:
#                             normal_index = class_to_idx[normal_name]
#                             mask[added_index, normal_index] = 1
#                     mask[added_index, added_index] = 1
#         else:
#             mask = np.zeros((cls_num, cls_num))
#             normal_mask = np.zeros(cls_num)
#
#         return mask, normal_mask
#
#     # def __getitem__(self, index: int):
#     #     """
#     #     Args:
#     #         index (int): Index
#     #
#     #     Returns:
#     #         tuple: (sample, target) where target is class_index of the target class.
#     #     """
#     #     path, target = self.samples[index]
#     #     sample = self.loader(path)
#     #     if self.transform is not None:
#     #         sample = self.transform(sample)
#     #     if self.target_transform is not None:
#     #         target = self.target_transform(target)
#     #
#     #     return sample, target
#     def __getitem__(self, index: int):
#         try:
#             path, target = self.samples[index]
#             sample = self.loader(path)
#             if self.transform is not None:
#                 sample = self.transform(sample)
#             if self.target_transform is not None:
#                 target = self.target_transform(target)
#         except:
#             return self.__getitem__(index + 1)
#         return sample, target

class DatasetLT_aug(datasets.ImageFolder):

    def __init__(
            self,
            root=None,
            transform=None,
            aug_data='no'
    ):
        super().__init__(
            root=root,
            transform=transform,
        )

        # self.descriptions = CLASSNAME2TEXT
        new_names = [""] * len(self.classes)
        self.cls_names = CLASSES_IMAGENET
        # for k, v in self.class_to_idx.items():
        #     original_id = int(k)
        #     new_id = int(v)
        #     new_names[new_id] = self.cls_names[original_id]
        self.cls_names = new_names

        self._tokenizer = SimpleTokenizer()

        self.descriptions_list = []
        # for cls_name in self.cls_names:
        #     self.descriptions_list.append(self.descriptions[cls_name])

        self.token_list = []
        # self.token_list = self.tokenize(self.descriptions_list)

        self.labels = []
        for index in range(len(self.samples)):
            path, target = self.samples[index]
            self.labels.append(target)

        self.labels = np.array(self.labels)


        self.idx_to_class = {}
        for clsname, idx in self.class_to_idx.items():
            self.idx_to_class[str(idx)] = clsname
        self.new_cls_num = [0] * len(self.idx_to_class)


        self.root = root

        self.conp_cls = self.root[:-len(self.root.split('/')[-1])] + "conp_cls.npy"

        self.added_clsname = []
        if os.path.exists(self.conp_cls):
            conp_cls = np.load(self.conp_cls, allow_pickle=True).item()
            for imagenet_cls, cls_list in conp_cls.items():
                self.added_clsname += cls_list

        self.aug_data = aug_data
        # if aug_data == 'yes':
        #     self.re_arrange()
        self.imgs = self.samples

        self.cls_num = self.get_cls_num()

        self.get_aug_info()
        self.class_indices = self.precompute_class_indices()
        self.resample_indices()
        #
        # print(self.indices)
        # print(self.indices)
        # sys.exit()


    def precompute_class_indices(self):
        # 预先计算并存储每个类别的索引
        # class_indices = {str(label): np.where(self.labels == label)[0] for label in range(len(self.cls_num))}
        class_indices = {}
        for label in tqdm(range(len(self.cls_num))):

            class_indices[str(label)] =  np.where(self.labels == label)[0]

        return class_indices

    def resample_indices(self):
        # 初始化索引列表
        indices = []
        # 对于每个类别
        for label in tqdm(range(len(self.cls_num))):
            # 找到这个类别的所有索引
            class_indices = self.class_indices[str(label)]
            # print(class_indices)
            # 如果类别是我们想要限制采样次数的类别
            cls_name = self.idx_to_class[str(label)]
            if cls_name in self.added_clsname:
                if len(class_indices) > self.max_count[label]:
                    # 从这个类别的索引中随机选择samples_per_class个
                    class_indices = np.random.choice(class_indices, int(self.max_count[label]), replace=True)
            # 添加到总索引列表中
            indices.extend(class_indices)

            self.new_cls_num[label] = len(class_indices)

        self.indices = indices


    def __len__(self):
        return len(self.indices)

    # def get_cls_num(self):
    #     cls_num = [0] * len(self.classes)
    #     for label in self.labels:
    #         cls_num[label] += 1
    #     return cls_num

    def get_cls_num(self):
        return self.new_cls_num

    def get_aug_info(self):
        wordname_to_num_path =  self.root[:-len(self.root.split('/')[-1])] + "wordname_to_num.npy"
        self.wordname_to_num = np.load(wordname_to_num_path, allow_pickle=True).item()

        self.max_count = np.zeros(len(self.idx_to_class))

        for cls_index in range(len(self.classes)):
            cls_name = self.idx_to_class[str(cls_index)]
            if cls_name in self.added_clsname:

                cls_num = self.wordname_to_num[cls_name]

                if cls_num > 100:
                    selected_num = 50
                elif cls_num > 20:
                    selected_num = cls_num
                else:
                    selected_num = 50
                self.max_count[cls_index] = selected_num
            else:
                self.max_count[cls_index] = self.cls_num[cls_index]


    def generate_mask(self, all_normal=True, mask_type=""):
        # return None, None
        class_to_idx = self.class_to_idx
        cls_num = len(class_to_idx)

        if self.aug_data == "no":
            mask = np.ones((cls_num, cls_num))
            normal_mask = np.ones(cls_num)

            return mask, normal_mask

        conp_cls_path = self.root[:-len(self.root.split('/')[-1])] + "conp_cls.npy"

        conp_cls = np.load(conp_cls_path, allow_pickle=True).item()

        added2normal = {}
        for imagenet_cls, cls_list in conp_cls.items():

            for added_name in cls_list:
                if added_name not in added2normal:
                    added2normal[added_name] = []
                if imagenet_cls not in added2normal[added_name]:
                    added2normal[added_name].append(imagenet_cls)

        normal_mask = np.zeros(cls_num)

        for cls_name, idx in class_to_idx.items():
            if 'n' not in cls_name:
                normal_mask[idx] = 1

        normal_idx = np.where(normal_mask == 1)[0]
        added_idx = np.where(normal_mask == 0)[0]
        cls_num_list = self.get_cls_num()

        if mask_type == "no_fm":

            mask = np.ones((cls_num, cls_num))

            for imagenet_cls, cls_list in conp_cls.items():

                cls_index = class_to_idx[imagenet_cls]
                if cls_num_list[cls_index] < 100:

                    for add_cls in cls_list:
                        if add_cls in class_to_idx:
                            add_index = class_to_idx[add_cls]
                            mask[cls_index, add_index] = 0

            for added_cls, normal_list in added2normal.items():
                if added_cls in class_to_idx:
                    added_index = class_to_idx[added_cls]

                    for normal_name in normal_list:
                        normal_index = class_to_idx[normal_name]
                        if cls_num_list[normal_index] < 100:
                            mask[added_index, normal_index] = 0

        elif mask_type == "no_fm_remove_normal_neg":

            mask = np.ones((cls_num, cls_num))

            for imagenet_cls, cls_list in conp_cls.items():

                cls_index = class_to_idx[imagenet_cls]
                # if cls_num_list[cls_index] < 100:

                for add_cls in cls_list:
                    if add_cls in class_to_idx:
                        add_index = class_to_idx[add_cls]
                        mask[cls_index, add_index] = 0

            for added_cls, normal_list in added2normal.items():
                if added_cls in class_to_idx:
                    added_index = class_to_idx[added_cls]

                    for normal_name in normal_list:
                        normal_index = class_to_idx[normal_name]
                        # if cls_num_list[normal_index] < 100:
                        mask[added_index, normal_index] = 0

        elif mask_type == "no_fm_remove_normal_neg_back":

            mask = np.ones((cls_num, cls_num))

            for imagenet_cls, cls_list in conp_cls.items():

                cls_index = class_to_idx[imagenet_cls]
                # if cls_num_list[cls_index] < 100:

                for add_cls in cls_list:
                    if add_cls in class_to_idx:
                        add_index = class_to_idx[add_cls]
                        mask[cls_index, add_index] = 0

            for added_cls, normal_list in added2normal.items():
                if added_cls in class_to_idx:
                    added_index = class_to_idx[added_cls]

                    mask[added_index, added_idx] = 0
                    mask[added_index, added_index] = 1

                    for normal_name in normal_list:
                        normal_index = class_to_idx[normal_name]
                        # if cls_num_list[normal_index] < 100:
                        mask[added_index, normal_index] = 0


        elif mask_type == "no_fm_1":

            mask = np.ones((cls_num, cls_num))

            for imagenet_cls, cls_list in conp_cls.items():

                cls_index = class_to_idx[imagenet_cls]
                if cls_num_list[cls_index] < 100:

                    for add_cls in cls_list:
                        if add_cls in class_to_idx:
                            add_index = class_to_idx[add_cls]
                            mask[cls_index, add_index] = 0

        elif mask_type == "no_fm_2":
            mask = np.ones((cls_num, cls_num))

            for added_cls, normal_list in added2normal.items():
                if added_cls in class_to_idx:
                    added_index = class_to_idx[added_cls]

                    for normal_name in normal_list:
                        normal_index = class_to_idx[normal_name]
                        if cls_num_list[normal_index] < 100:
                            mask[added_index, normal_index] = 0


        elif mask_type == "no_fm_2_back":
            mask = np.ones((cls_num, cls_num))

            for added_cls, normal_list in added2normal.items():
                if added_cls in class_to_idx:
                    added_index = class_to_idx[added_cls]

                    mask[added_index, added_idx] = 0
                    mask[added_index, added_index] = 1

                    for normal_name in normal_list:
                        normal_index = class_to_idx[normal_name]
                        if cls_num_list[normal_index] < 100:
                            mask[added_index, normal_index] = 0

        elif mask_type == "no_fm_2_back_remove_normal":
            mask = np.ones((cls_num, cls_num))

            for added_cls, normal_list in added2normal.items():
                if added_cls in class_to_idx:
                    added_index = class_to_idx[added_cls]

                    mask[added_index, added_idx] = 0
                    mask[added_index, added_index] = 1

                    mask[added_index, normal_idx] = 0

                    # for normal_name in normal_list:
                    #     normal_index = class_to_idx[normal_name]
                    #     if cls_num_list[normal_index] < 100:
                    #         mask[added_index, normal_index] = 0

        elif mask_type == "no_fm_2_remove_normal_neg_back":
            # print("no_fm_2_remove_normal_neg")
            # sys.exit()
            mask = np.ones((cls_num, cls_num))

            for added_cls, normal_list in added2normal.items():
                if added_cls in class_to_idx:
                    added_index = class_to_idx[added_cls]

                    mask[added_index, added_idx] = 0
                    mask[added_index, added_index] = 1

                    for normal_name in normal_list:
                        normal_index = class_to_idx[normal_name]
                        # if cls_num_list[normal_index] < 100:
                        mask[added_index, normal_index] = 0

        elif mask_type == "no_fm_2_remove_normal_neg":
            # print("no_fm_2_remove_normal_neg")
            # sys.exit()
            mask = np.ones((cls_num, cls_num))

            for added_cls, normal_list in added2normal.items():
                if added_cls in class_to_idx:
                    added_index = class_to_idx[added_cls]

                    for normal_name in normal_list:
                        normal_index = class_to_idx[normal_name]
                        # if cls_num_list[normal_index] < 100:
                        mask[added_index, normal_index] = 0
        elif mask_type == "normal":

            cls_num = len(class_to_idx)

            mask = np.zeros((cls_num, cls_num))

            normal_mask = np.zeros(cls_num)

            for cls_name, idx in class_to_idx.items():
                if 'n' not in cls_name:
                    normal_mask[idx] = 1

            normal_idx = np.where(normal_mask == 1)[0]
            added_idx = np.where(normal_mask == 0)[0]

            for imagenet_cls, cls_list in conp_cls.items():

                cls_index = class_to_idx[imagenet_cls]
                mask[cls_index, normal_idx] = 1

                for add_cls in cls_list:
                    if add_cls in class_to_idx:
                        add_index = class_to_idx[add_cls]
                        mask[cls_index, add_index] = 1

            for added_cls, normal_list in added2normal.items():
                if added_cls in class_to_idx:
                    added_index = class_to_idx[added_cls]

                    if all_normal:
                        mask[added_index, normal_idx] = 1
                    else:
                        for normal_name in normal_list:
                            normal_index = class_to_idx[normal_name]
                            mask[added_index, normal_index] = 1
                    mask[added_index, added_index] = 1
        else:
            mask = np.zeros((cls_num, cls_num))
            normal_mask = np.zeros(cls_num)

        return mask, normal_mask


    def __getitem__(self, index: int):
        try:

            idx = self.indices[index]

            path, target = self.samples[idx]

            sample = self.loader(path)
            if self.transform is not None:
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)
        except:
            return self.__getitem__(index + 1)
        return sample, target

if __name__ == '__main__':
    dataset_train = DatasetLT(os.path.join('/diskC/xzz/ImageNet-LT', 'train'), transform=None)
    cls_num = dataset_train.get_cls_num()
    print(cls_num)
