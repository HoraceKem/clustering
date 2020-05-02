import numpy as np


class KMeansPlus:
    def __init__(self, cluster_num, sample_num_all, max_iter=100):
        self.cluster_num = cluster_num
        self.max_iter = max_iter
        self.sample_num_all = sample_num_all
        self.cluster_centers = None

    def generate_init_center(self, data):
        idx = np.random.randint(0, self.sample_num_all)
        self.cluster_centers = [data[idx]]

        for i in range(self.cluster_num-1):
            dist = np.zeros((self.sample_num_all, len(self.cluster_centers)))
            for center_idx in range(len(self.cluster_centers)):
                dist[:, center_idx] = np.linalg.norm(
                    data - self.cluster_centers[center_idx], axis=1)
            dist = np.min(dist, axis=1)
            p = dist / np.sum(dist)
            next_center_idx = np.random.choice(self.sample_num_all, p=p)
            self.cluster_centers.append(data[next_center_idx])
        self.cluster_centers = np.array(self.cluster_centers)

    def find_nearest_center(self, point):
        min_dist = float('inf')
        label = 0
        for i in range(self.cluster_num):
            dist = np.linalg.norm(point-self.cluster_centers[i])
            if dist < min_dist:
                min_dist = dist
                label = i
        return label

    def fit(self, data):
        self.generate_init_center(data)
        label_all = []
        for it in range(self.max_iter):
            label_all = []
            for id in range(self.sample_num_all):
                label = self.find_nearest_center(data[id])
                label_all.append(label)
            label_all = np.array(label_all)
            for i in range(self.cluster_num):
                idx = np.argwhere(label_all == i)
                points = np.vstack([data[idx]])
                self.cluster_centers[i] = np.mean(points, axis=0)
        return label_all, self.cluster_centers
