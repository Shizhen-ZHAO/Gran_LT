import os

from classes_aug import CLASSES_INAT, CLASSES_Places, CLASSES, CUSTOM_TEMPLATES
import torch
from clip import load as clip_load
import torch.nn as nn
from torchvision import datasets, transforms
import clip
from tqdm import tqdm
import pickle

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        self.token_embedding = clip_model.token_embedding

    def forward(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

templates = CUSTOM_TEMPLATES['ImageNet']
model, preprocess = clip_load('ViT-B/16', "cpu")

text_model = TextEncoder(model)

text_model = torch.nn.DataParallel(text_model, device_ids = [0, 1, 2, 3])
text_model.to(f'cuda:{text_model.device_ids[0]}')

for name, p in text_model.named_parameters():
    p.requires_grad = False

embedding_save_root = "./clip_text_embedding"



def read_text(path):
    file1 = open(path, 'r')
    Lines = file1.readlines()

    name_list = []
    for line in Lines:
        name_list.append(line[:-1])

    return name_list

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
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target



def imagenet_aug_name():

    imagenet_path = "/mnt/sda/zsz/data/imagenetLT_auxiliary/all_au/train"
    id_path = "/home/szzhao/LT_project/vit_LT/process_granularity/imagenet_22k_clsname/imagenet21k_wordnet_ids.txt"
    name = "/home/szzhao/LT_project/vit_LT/process_granularity/imagenet_22k_clsname/imagenet21k_lemmas.txt"

    id_list = read_text(id_path)
    name_list = read_text(name)
    id2name = {}

    for idx in range(len(id_list)):
        id2name[id_list[idx]] = name_list[idx]

    fname_to_rname = {}
    fake_names = os.listdir(imagenet_path)

    for fname in fake_names:
        if 'n' in fname:
            fname_to_rname[fname] = id2name[fname]
        else:
            fname_to_rname[fname] = CLASSES[int(fname)]

    dataset = DatasetLT(imagenet_path)
    new_names = ['']*len(fake_names)
    for k, v in dataset.class_to_idx.items():
        new_id = int(v)
        print(k)
        new_names[new_id] = fname_to_rname[str(k)]

    classifier_weight_list = []

    for cls_name in tqdm(new_names):
        texts = torch.cat([clip.tokenize(templates.format(c)) for c in [cls_name]])
        texts = texts.to(f'cuda:{text_model.device_ids[0]}')
        zeroshot_weight = text_model(texts).float().detach()
        classifier_weight_list.append(zeroshot_weight)

    classifier_weights = torch.cat(classifier_weight_list, dim=0).cpu().numpy()
    print(classifier_weights.shape)

    save_path = embedding_save_root + '/' + "classifier_weights_imagenetLT_aug.pkl"
    file = open(save_path, 'wb')
    pickle.dump(classifier_weights, file)








def place_aug_name():

    place_path = "/mnt/sdc/zsz/aug_images/places_exp/exp_2/train"

    id_path = "/home/szzhao/LT_project/vit_LT/process_granularity/imagenet_22k_clsname/imagenet21k_wordnet_ids.txt"
    name = "/home/szzhao/LT_project/vit_LT/process_granularity/imagenet_22k_clsname/imagenet21k_lemmas.txt"

    id_list = read_text(id_path)
    name_list = read_text(name)
    id2name = {}

    for idx in range(len(id_list)):
        id2name[id_list[idx]] = name_list[idx]

    fname_to_rname = {}
    fake_names = os.listdir(place_path)

    for fname in fake_names:
        if fname in id2name:
            fname_to_rname[fname] = id2name[fname]
        elif fname.isdigit():
            fname_to_rname[fname] = CLASSES_Places[int(fname)]
        else:
            fname_to_rname[fname] = fname

    dataset = DatasetLT(place_path)
    new_names = [''] * len(fake_names)
    for k, v in dataset.class_to_idx.items():
        new_id = int(v)
        print(k)
        new_names[new_id] = fname_to_rname[str(k)]

    classifier_weight_list = []

    for cls_name in tqdm(new_names):
        texts = torch.cat([clip.tokenize(templates.format(c)) for c in [cls_name]])
        texts = texts.to(f'cuda:{text_model.device_ids[0]}')
        zeroshot_weight = text_model(texts).float().detach()
        classifier_weight_list.append(zeroshot_weight)

    classifier_weights = torch.cat(classifier_weight_list, dim=0).cpu().numpy()
    print(classifier_weights.shape)

    save_path = embedding_save_root + '/' + "classifier_weights_placeLT_aug.pkl"
    file = open(save_path, 'wb')
    pickle.dump(classifier_weights, file)



def inat_aug_name():

    inat_path = "/home/szzhao/LT_project/vit_LT/data/iNat18/train"

    id_path = "/home/szzhao/LT_project/vit_LT/process_granularity/imagenet_22k_clsname/imagenet21k_wordnet_ids.txt"
    name = "/home/szzhao/LT_project/vit_LT/process_granularity/imagenet_22k_clsname/imagenet21k_lemmas.txt"

    id_list = read_text(id_path)
    name_list = read_text(name)
    id2name = {}

    for idx in range(len(id_list)):
        id2name[id_list[idx]] = name_list[idx]

    fname_to_rname = {}
    fake_names = os.listdir(inat_path)

    for fname in fake_names:
        if fname in id2name:
            fname_to_rname[fname] = id2name[fname]
        elif fname.isdigit():
            fname_to_rname[fname] = CLASSES_INAT[int(fname)]
        else:
            fname_to_rname[fname] = fname

    dataset = DatasetLT(inat_path)
    new_names = [''] * len(fake_names)
    for k, v in dataset.class_to_idx.items():
        new_id = int(v)
        print(k)
        new_names[new_id] = fname_to_rname[str(k)]

    classifier_weight_list = []

    for cls_name in tqdm(new_names):
        texts = torch.cat([clip.tokenize(templates.format(c)) for c in [cls_name]])
        texts = texts.to(f'cuda:{text_model.device_ids[0]}')
        zeroshot_weight = text_model(texts).float().detach()
        classifier_weight_list.append(zeroshot_weight)

    classifier_weights = torch.cat(classifier_weight_list, dim=0).cpu().numpy()
    print(classifier_weights.shape)

    save_path = embedding_save_root + '/' + "iNat18.pkl"
    file = open(save_path, 'wb')
    pickle.dump(classifier_weights, file)


def initial_classifier(cls_names):
    with torch.no_grad():
        classnames = cls_names
        templates = CUSTOM_TEMPLATES['ImageNet']

        model, preprocess = clip_load('ViT-B/16', "cpu")
        torch.cuda.empty_cache()

        text_model = TextEncoder(model)

        text_model = torch.nn.DataParallel(text_model, device_ids = [0, 1, 2, 3])
        text_model.to(f'cuda:{text_model.device_ids[0]}')
        # self.text_model.to('cuda')
        # self.text_model.eval()

        for name, p in text_model.named_parameters():
            p.requires_grad = False

        texts = torch.cat([clip.tokenize(templates.format(c)) for c in classnames])
        # texts = texts.cuda()
        texts = texts.to(f'cuda:{text_model.device_ids[0]}')
        # texts = texts.to('cuda')
        zeroshot_weights = text_model(texts).float().detach()
        torch.cuda.empty_cache()

        texts = texts[:10]
        zeroshot_weights2 = text_model(texts[:10]).float().detach()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()

        texts.cpu()
        text_model.cpu()

    return zeroshot_weights


if __name__ == '__main__':
    inat_aug_name()








