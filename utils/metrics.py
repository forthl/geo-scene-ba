import numpy as np
from numpy.random import uniform
import random


def distance(p, data):
    """
    p: (m,)
    data: (n,m)
    output: (n,)
    """
    return np.sqrt(np.sum((p - data) ** 2, axis=1))


class KMeans:
    def __init__(self, k_clusters=5, max_iter=200):
        self.k_clusters = k_clusters
        self.max_iter = max_iter

    def fit(self, data):
        # initialize centroids using kmeans++
        self.centroids = [random.choice(data)]
        for _ in range(self.k_clusters - 1):
            # distances from datapoints to centroids
            distances = np.sum([distance(centroid, data) for centroid in self.centroids], axis=0)
            # normalize dists
            distances /= np.sum(distances)
            # choose centroid
            new_centroid_idx, = np.random.choice(range(len(data)), size=1, p=distances)
            # add centroid
            self.centroids += [data[new_centroid_idx]]

        # initialize fitting
        iter = 0
        prev_centroids = None

        # stop if centroids converged or when it reached max_iter
        while np.not_equal(self.centroids, prev_centroids).any() and iter < self.max_iter:
            # sort and assign embeddings to centroids
            sorted_points = [[] for _ in range(self.k_clusters)]
            for embedding in data:
                distances = distance(embedding, self.centroids)
                centroid_idx = np.argmin(distances)
                sorted_points[centroid_idx].append(embedding)

            # manage centroids
            prev_centroids = self.centroids
            self.centroids = [np.mean(cluster, axis=0) for cluster in sorted_points]

            # check for empty clusters and reassign previous centroid
            for i, centroid in enumerate(self.centroids):
                if np.isnan(centroid).any():
                    self.centroids[i] = prev_centroids[i]
            iter += 1

    def evaluate(self, data):
        """
            data: (n,m) - stores datapoints
            centroids: (n,m) - stores centroids per datapoint
            centroid_idxs: (n,) - stores indices of centroids per datapoint
            """
        centroids = []
        centroid_idxs = []
        for x in data:
            dists = distance(x, self.centroids)
            centroid_idx = np.argmin(dists)
            centroids.append(self.centroids[centroid_idx])
            centroid_idxs.append(centroid_idx)

        print(data)
        print("Check\n\n\n\n\n")
        print(centroids)
        print("Check\n\n\n\n\n")
        print(centroid_idxs)

        return centroids, centroid_idxs
