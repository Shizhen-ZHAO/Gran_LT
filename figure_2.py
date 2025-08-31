import numpy as np
import matplotlib.pyplot as plt
import os

def plot_figure(figure_name=''):


    root = "/home/szzhao/LT_project/vit_LT/figure"
    info_save_path = os.path.join(root, figure_name, 'data_info.npy')

    data_info = np.load(info_save_path, allow_pickle=True).item()

    features = data_info["features"]
    labels = data_info["labels"]
    data_embedded = data_info["data_embedded"]
    few_name = data_info["few_name"]
    many_name = data_info["many_name"]
    class_to_idx = data_info["class_to_idx"]
    unique_label = data_info["unique_label"]


    fig = plt.figure()

    colors_index = -1

    markersize = 40

    colors = ["#02c39a", "#fca7a7"]
    for one_label in unique_label:
        # print(one_label)
        label1_index = np.where(labels == one_label)[0]
        if one_label == class_to_idx[str(few_name)]:
            # cls_name = CLASSES[1]
            # cls_name = cls_name.split(' ')
            # cls_name = cls_name[0].capitalize() + " " + cls_name[1].capitalize()
            plt.scatter(data_embedded[label1_index, 0], data_embedded[label1_index, 1], s=markersize, alpha=1, c="#ff7c38",
                        label="Irish Wolfhound")
        elif one_label == class_to_idx[str(many_name)]:
            # cls_name = CLASSES[2]
            # cls_name = cls_name.split(' ')
            # cls_name = cls_name[0].capitalize() + " " + cls_name[1].capitalize()
            plt.scatter(data_embedded[label1_index, 0], data_embedded[label1_index, 1], s=markersize, alpha=1, c="#216583",
                        label="Scottish Deerhound")

        else:
            if figure_name == 'figure_d':
                colors_index += 1
                plt.scatter(data_embedded[label1_index, 0], data_embedded[label1_index, 1], s=markersize, alpha=1,
                            c=colors[colors_index],
                            label=str(one_label))

    save_fig_path = os.path.join(root, figure_name + ".pdf")

    plt.xticks([])
    plt.yticks([])
    plt.savefig(save_fig_path)

plot_figure('figure_a')
plot_figure('figure_b')
plot_figure('figure_c')
plot_figure('figure_d')
