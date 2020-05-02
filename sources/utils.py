import termcolor
import numpy as np
import matplotlib.pyplot as plt


def toRed(content): return termcolor.colored(content, "red", attrs=["bold"])


def toBlue(content): return termcolor.colored(content, "blue", attrs=["bold"])


def toGreen(content): return termcolor.colored(content, "green", attrs=["bold"])


def evaluate(data, predicted_label):
    if not len(predicted_label) == 3:
        silhouette_score = 0
        for i in range(len(data)):
            idx_in = np.argwhere(predicted_label == predicted_label[i])
            idx_out = np.argwhere(predicted_label != predicted_label[i])
            points_in = np.vstack(data[idx_in])
            points_out = np.vstack(data[idx_out])
            dist_in, dist_out = 0, 0
            for j in range(points_in.shape[0]):
                dist_in += np.linalg.norm(data[i]-points_in[j])
            for k in range(points_out.shape[0]):
                dist_out += np.linalg.norm(data[i]-points_out[k])
            dist_in = dist_in / (points_in.shape[0]-1)
            dist_out = dist_out / (points_out.shape[0])
            silhouette_score += (dist_out - dist_in) / max(dist_in, dist_out)
        silhouette_score = silhouette_score / len(data)
    else:
        silhouette_score = [0, 0, 0]
        for m in range(3):
            for i in range(len(data)):
                idx_in = np.argwhere(predicted_label[m] == predicted_label[m][i])
                idx_out = np.argwhere(predicted_label[m] != predicted_label[m][i])
                points_in = np.vstack(data[idx_in])
                points_out = np.vstack(data[idx_out])
                dist_in, dist_out = 0, 0
                for j in range(points_in.shape[0]):
                    dist_in += np.linalg.norm(data[i] - points_in[j])
                for k in range(points_out.shape[0]):
                    dist_out += np.linalg.norm(data[i] - points_out[k])
                dist_in = dist_in / (points_in.shape[0] - 1)
                dist_out = dist_out / (points_out.shape[0])
                silhouette_score[m] += (dist_out - dist_in) / max(dist_in, dist_out)
            silhouette_score[m] = silhouette_score[m] / len(data)
    return silhouette_score


def visualize_single(cluster_num, data, predicted_label, centers):
    plt.figure(figsize=(5.3, 5))
    for j in range(cluster_num):
        idx = np.argwhere(predicted_label == j)
        points = np.vstack(data[idx])
        f1, = plt.plot(points[:, 0], points[:, 1], 'o', markersize=5)
        f2, = plt.plot(centers[j, 0], centers[j, 1], 'X', markersize=10)
    plt.legend([f1, f2], ['points', 'cluster center'], loc='upper right', scatterpoints=1)
    plt.show()


def visualize_multi(cluster_num, data, predicted_label, centers, score, time):
    name = ['K-Means', 'K-Means++', 'VQ_LBG']
    fig, axs = plt.subplots(2, 3, figsize=(17, 10))
    for i in range(len(predicted_label)):
        # sub_figure = plt.subplot(1, len(predicted_label), i+1)
        for j in range(cluster_num):
            idx = np.argwhere(predicted_label[i] == j)
            points = np.vstack(data[idx])
            f1, = axs[0, i].plot(points[:, 0], points[:, 1], 'o', markersize=5)
            f2, = axs[0, i].plot(centers[i][j, 0], centers[i][j, 1], 'X', markersize=10)
        axs[0, i].set_title(name[i])
        axs[0, i].legend([f1, f2], ['points', 'cluster center'], loc='upper right', scatterpoints=1)
    offset = 3 * min(score) - 2 * max(score)
    score[:] = [x-offset for x in score]
    axs[1, 0].barh(name, score, left=offset)
    axs[1, 0].set_title("score")
    axs[1, 1].barh(name,time)
    axs[1, 1].set_title("time")
    plt.show()
