import sys

import matplotlib.pyplot as plt
import matplotlib
import os
from util.datasets import build_dataset, build_dataset_place_test, build_dataset_train
from util.datasets import DatasetLT, DatasetLT_aug
import numpy as np

def sort_and_plot(list1, list2, dataset_name):
    # Sort the lists in descending order
    sorted_list1 = sorted(list1, reverse=True)
    sorted_list2 = sorted(list2, reverse=True)

    # Import matplotlib for plotting

    matplotlib.rcParams['font.family'] = 'Arial'



    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    # Plotting the sorted lists
    plt.figure(figsize=(8, 5))

    #############

    indices_list1 = [i - 10 for i in range(len(sorted_list1))]  # Offset for list 1
    indices_list2 = [i + int(len(sorted_list1)/6) for i in range(len(sorted_list2))]  # Offset for list 2

    #############

    legend1 = dataset_name
    legend2 = dataset_name + "+ Auxiliary"

    # Plotting both lists with their index on the x-axis and their value on the y-axis
    plt.plot(indices_list1, sorted_list1, 'o-', label=legend1, linewidth=3)
    plt.plot(indices_list2, sorted_list2, 's-', label=legend2, linewidth=3)

    # Adding titles and labels
    # plt.title('Sorted Lists Plot')
    plt.xlabel('Category Index', fontsize=16)
    plt.ylabel('Number of Sample', fontsize=16)

    plt.xticks(size=10)
    plt.yticks(size=10)


    # Showing the legend
    plt.legend(fontsize=16)


    save_fig_path = os.path.join("/home/szzhao/LT_project/vit_LT/distribution_plot", dataset_name+".pdf")


    plt.savefig(save_fig_path)



# def sort_and_plot_1(list1, list2, dataset_name):
#     # Sort the lists in descending order
#     sorted_list1 = sorted(list1, reverse=True)
#     sorted_list2 = sorted(list2, reverse=True)
#
#
#     matplotlib.rcParams['font.family'] = 'Arial'
#
#     # Plotting the sorted lists
#     # plt.figure(figsize=(8, 5))
#
#     indices_list1 = [i for i in range(len(sorted_list1))]  # Offset for list 1
#     indices_list2 = [i + int(len(sorted_list1)/6) for i in range(len(sorted_list2))]  # Offset for list 2
#
#     legend1 = dataset_name
#     legend2 = dataset_name + "+ Auxiliary"
#
#     fig, ax1 = plt.subplots(figsize=(8, 5))
#
#
#     ax1.plot(indices_list1, sorted_list1, 'o-',  label=legend1, linewidth=3, color='Blue')
#     ax1.set_xlabel('Category Index', color='blue', fontsize=16)
#     ax1.tick_params(axis='x', labelcolor='blue')
#     min_length = min(len(sorted_list1), len(sorted_list2))
#     tick_interval = max(1, min_length // 4)
#
#
#     ax1.set_xlim(-50, max(len(sorted_list1), len(sorted_list2)))
#     ax1.set_xticks(np.arange(len(sorted_list1))[::tick_interval])
#
#
#     ax2 = ax1.twiny()
#
#     ax2.plot(indices_list2, sorted_list2, 's-', label=legend2, linewidth=3, color='red')
#     ax2.set_xlabel('Category Index', color='red', fontsize=16)
#     ax2.tick_params(axis='x', labelcolor='red')
#
#     ax2.set_xlim(ax1.get_xlim())
#     # ax2.set_xticks(range(int(len(sorted_list1)/6), len(sorted_list2) + int(len(sorted_list1)/6)))
#
#
#     plt.ylabel('Number of Sample', fontsize=16)
#
#
#     plt.legend(fontsize=16)
#
#     save_fig_path = os.path.join("/home/szzhao/LT_project/vit_LT/distribution_plot", dataset_name+".jpg")
#
#
#     plt.savefig(save_fig_path)


#
# def sort_and_plot_all(list1, list2, list3, list4, list5, list6):
#     # Sort the lists in descending order
#     sorted_list1 = sorted(list1, reverse=True)
#     sorted_list2 = sorted(list2, reverse=True)
#
#     sorted_list3 = sorted(list3, reverse=True)
#     sorted_list4 = sorted(list4, reverse=True)
#
#     sorted_list5 = sorted(list5, reverse=True)
#     sorted_list6 = sorted(list6, reverse=True)
#
#     # Import matplotlib for plotting
#
#     matplotlib.rcParams['font.family'] = 'Arial'
#
#     fig, axes = plt.subplots(1, 3, figsize=(24, 5))
#     # Plotting the sorted lists
#
#     #############
#
#     indices_list1 = [i - 10 for i in range(len(sorted_list1))]  # Offset for list 1
#     indices_list2 = [i + int(len(sorted_list1)/6) for i in range(len(sorted_list2))]  # Offset for list 2
#
#     #############
#
#     legend1 = "ImageNet-LT"
#     legend2 = "ImageNet-LT" + "+ Auxiliary"
#
#     # Plotting both lists with their index on the x-axis and their value on the y-axis
#     axes[0].plot(indices_list1, sorted_list1, 'o-', label=legend1, linewidth=3)
#     axes[0].plot(indices_list2, sorted_list2, 's-', label=legend2, linewidth=3)
#
#     # Adding titles and labels
#     # plt.title('Sorted Lists Plot')
#     axes[0].set_xlabel('Category Index', fontsize=16)
#     axes[0].set_ylabel('Number of Sample', fontsize=16)
#
#     # axes[0].set_xticks(size=10)
#     # axes[0].set_yticks(size=10)
#
#
#     # Showing the legend
#     axes[0].legend(fontsize=16)
#
#
#
#     save_fig_path = os.path.join("/home/szzhao/LT_project/vit_LT/distribution_plot", 'dataset_name'+".pdf")
#
#
#     plt.savefig(save_fig_path)

if __name__ == '__main__':

    sample_num_dict = {}

    dataset_path = "/home/szzhao/LT_project/vit_LT/data/ImageNet-LT/train"
    aug_dataset_path = "/mnt/sda/zsz/data/imagenetLT_auxiliary/all_au/train"

    dataset_train = DatasetLT(dataset_path)
    dataset_train_aug = DatasetLT(aug_dataset_path, aug_data='yes')

    cls_num_train_imagenet = dataset_train.get_cls_num()
    cls_num_train_aug_imagenet = dataset_train_aug.get_cls_num()

    sample_num_dict['imagenet'] = cls_num_train_imagenet
    sample_num_dict['imagenet_aug'] = cls_num_train_aug_imagenet

    sort_and_plot(cls_num_train_imagenet, cls_num_train_aug_imagenet, "ImageNet-LT")



    dataset_path = "/home/szzhao/LT_project/vit_LT/data/Place/train"
    aug_dataset_path = "/mnt/sdc/zsz/aug_images/places_exp/exp_2/train"

    dataset_train = DatasetLT(dataset_path)
    dataset_train_aug = DatasetLT(aug_dataset_path, aug_data='yes')

    cls_num_train_place = dataset_train.get_cls_num()
    cls_num_train_aug_place = dataset_train_aug.get_cls_num()

    sample_num_dict['place'] = cls_num_train_place
    sample_num_dict['place_aug'] = cls_num_train_aug_place

    sort_and_plot(cls_num_train_place, cls_num_train_aug_place, "Place-LT")

    # sort_and_plot_all(cls_num_train_imagenet, cls_num_train_aug_imagenet, cls_num_train_place, cls_num_train_aug_place, cls_num_train_place, cls_num_train_aug_place)
    # sys.exit()


    dataset_path = "/home/szzhao/LT_project/vit_LT/data/iNat18/train"
    aug_dataset_path = "/mnt/sda/zsz/iNat18/train"

    dataset_train = DatasetLT(dataset_path)
    dataset_train_aug = DatasetLT_aug(aug_dataset_path, aug_data='yes')


    cls_num_train_inat = dataset_train.get_cls_num()
    cls_num_train_aug_inat = dataset_train_aug.new_cls_num

    sample_num_dict['inat'] = cls_num_train_inat
    sample_num_dict['inat_aug'] = cls_num_train_aug_inat

    # sort_and_plot_all(cls_num_train_imagenet, cls_num_train_aug_imagenet, cls_num_train_place, cls_num_train_aug_place, cls_num_train_inat, cls_num_train_aug_inat)

    sort_and_plot(cls_num_train_inat, cls_num_train_aug_inat, "iNat18")




    # import json
    # save_path = "/home/szzhao/LT_project/vit_LT/distribution_plot/distribution_info"
    # with open(save_path, "w") as f:
    #     json.dump(sample_num_dict, f)
    #


