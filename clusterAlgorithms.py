import numpy as np
from sklearn.cluster import KMeans
from kneed import KneeLocator
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt


class Kmeans:
    def __init__(self, data, max_k=20, optimal_k_method='e'):
        self.data = np.transpose(data)
        self.max_k = max_k
        self.optimal_k_method = optimal_k_method

    def optimal_k_elbow(self):
        distortions = []
        for k in range(1, self.max_k + 1):
            kmeans = KMeans(n_clusters=k, init='k-means++', random_state=0, n_init='auto').fit(self.data)
            distortions.append(kmeans.inertia_)

        # Find the optimal k
        kn = KneeLocator(range(1, self.max_k + 1), distortions, curve='convex', direction='decreasing')
        return kn.knee

    def optimal_k_bic(self):
        # TODO
        return 1

    def optimal_k_ml(self):
        # TODO
        return 2

    def optimal_k_vrc(self):
        # TODO
        return 3

    def find_optimal_k(self):
        if self.optimal_k_method == 'e':
            optimal_k = self.optimal_k_elbow()
        elif self.optimal_k_method == 'bic':
            optimal_k = self.optimal_k_bic()
        elif self.optimal_k_method == 'ml':
            optimal_k = self.optimal_k_ml()
        elif self.optimal_k_method == 'vrc':
            optimal_k = self.optimal_k_vrc()
        else:
            optimal_k = 0
        return optimal_k

    def kmeans_clustering(self, k):
        kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto').fit(self.data)
        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_
        return labels, centroids

    def find_clusters(self):
        optimal_k = self.find_optimal_k()
        print(f"Optimal k: {optimal_k}")
        labels, centroids = self.kmeans_clustering(optimal_k)
        return labels, centroids, self.data


class GaussianMixtureModel:

    def __init__(self, data, max_k=20, optimal_k_method='bic'):
        self.data = np.transpose(data)
        self.max_k = max_k
        self.optimal_k_method = optimal_k_method

    def optimal_k_elbow(self):
        # TODO
        return 1

    def optimal_k_bic(self):
        ks = np.arange(1, self.max_k)
        bics = []
        for k in ks:
            gmm = GaussianMixture(n_components=k, init_params='kmeans')
            gmm.fit(self.data)
            bics.append(gmm.bic(self.data))

        # Plot the data
        fig, ax = plt.subplots()
        ax.plot(ks, bics)
        ax.set_xlabel(r'Number of clusters, $k$')
        ax.set_ylabel('BIC')
        ax.set_xticks(ks);

        diff = [x - bics[i - 1] for i, x in enumerate(bics)][1:]

        return diff.index(min(diff)) + 2

    def optimal_k_ml(self):
        # TODO
        return 2

    def optimal_k_vrc(self):
        # TODO
        return 3

    def find_optimal_k(self):
        if self.optimal_k_method == 'e':
            optimal_k = self.optimal_k_elbow()
        elif self.optimal_k_method == 'bic':
            optimal_k = self.optimal_k_bic()
        elif self.optimal_k_method == 'ml':
            optimal_k = self.optimal_k_ml()
        elif self.optimal_k_method == 'vrc':
            optimal_k = self.optimal_k_vrc()
        else:
            optimal_k = 0
        return optimal_k

    def gmm_clustering(self, k):
        gmm = GaussianMixture(n_components=k).fit(self.data)

        # data points assigned to a cluster
        labels = gmm.predict(self.data)
        centroids = np.zeros((3, 3))
        return labels, centroids

    def find_clusters(self):
        optimal_k = self.find_optimal_k()
        print(f"Optimal k: {optimal_k}")
        labels, centroids = self.gmm_clustering(optimal_k)
        return labels, centroids, self.data
