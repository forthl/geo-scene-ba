import numpy as np
from sklearn.cluster import KMeans
from kneed import KneeLocator
from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm


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

    def find_optimal_k(self):
        if self.optimal_k_method == 'e':
            optimal_k = self.optimal_k_elbow()
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


class Spectral:

    def __init__(self, data, max_k=20, optimal_k_method='sil'):
        self.data = np.transpose(data)
        self.max_k = max_k
        self.optimal_k_method = optimal_k_method

    def optimal_k_sil(self):
        ks = np.arange(2, self.max_k)
        silhouette_scores = []
        best_score = -1
        optimal_k = -1

        for k in ks:
            spectral = SpectralClustering(n_clusters=k, affinity='nearest_neighbors')
            labels = spectral.fit_predict(self.data)
            score = silhouette_score(self.data, labels)
            silhouette_scores.append(score)

            if score > best_score:
                best_score = score
                optimal_k = k

        plt.plot(ks, silhouette_scores, marker='o')
        plt.xlabel("Number of Clusters")
        plt.ylabel("Silhouette Score")
        plt.title("Silhouette Score vs Number of Clusters")
        plt.show()

        return optimal_k

    def find_optimal_k(self):
        if self.optimal_k_method == 'sil':
            optimal_k = self.optimal_k_sil()
        else:
            optimal_k = 0
        return optimal_k

    def spectral_clustering(self, k):
        spectral = SpectralClustering(n_clusters=k, affinity='nearest_neighbors')

        # data points assigned to a cluster
        labels = spectral.fit_predict(self.data)
        centroids = np.zeros((3, 3))
        return labels, centroids

    def find_clusters(self):
        optimal_k = self.find_optimal_k()
        print(f"Optimal k: {optimal_k}")
        labels, centroids = self.spectral_clustering(optimal_k)
        return labels, centroids, self.data


class GaussianMixtureModel:

    def __init__(self, data, max_k=20, optimal_k_method='bic', n_init=5,
                 covariance_type='full', init_params='k-means++'):
        self.data = np.transpose(data)
        self.max_k = max_k
        self.optimal_k_method = optimal_k_method
        self.n_init = n_init
        self.covariance_type = covariance_type
        self.init_params = init_params

    def optimal_k_elbow(self):
        # TODO
        return 1

    def optimal_k_bic(self):
        ks = np.arange(1, self.max_k)
        bics = []
        for k in ks:
            gmm = GaussianMixture(n_components=k, n_init=self.n_init, covariance_type=self.covariance_type,
                                  init_params=self.init_params, max_iter=1000)
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

    def find_optimal_k(self):
        if self.optimal_k_method == 'bic':
            optimal_k = self.optimal_k_bic()
        else:
            optimal_k = 0
        return optimal_k

    def gmm_clustering(self, k):
        gmm = GaussianMixture(n_components=k, n_init=self.n_init, covariance_type=self.covariance_type,
                              init_params=self.init_params, max_iter=1000).fit(self.data)

        # data points assigned to a cluster
        labels = gmm.predict(self.data)
        centroids = np.zeros((3, 3))
        return labels, centroids

    def find_clusters(self):
        optimal_k = self.find_optimal_k()
        print(f"Optimal k: {optimal_k}")
        labels, centroids = self.gmm_clustering(optimal_k)
        return labels, centroids, self.data

class BayesianGaussianMixtureModel:
    def __init__(self, data, max_k=20, n_init=5,
                 covariance_type='full', init_params='k-means++'):
        self.data = np.transpose(data)
        self.max_k = max_k
        self.n_init = n_init
        self.covariance_type = covariance_type
        self.init_params = init_params

    def find_optimal_k(self):
        bgmm = BayesianGaussianMixture(n_components=self.max_k, covariance_type=self.covariance_type,
                                      init_params=self.init_params)
        bgmm.fit(self.data)
        bgmm_weights = bgmm.weights_
        optimal_k = (np.round(bgmm_weights, 2) > 0.05).sum()

        # Plot the data
        fig, ax = plt.subplots()
        ax.plot(range(len(bgmm_weights)), bgmm_weights)
        ax.set_xlabel(r'Number of clusters, $k$')
        ax.set_ylabel('weights')
        ax.set_xticks(range(len(bgmm_weights)))

        print('Estimated number of clusters: ' + str(optimal_k))

        return optimal_k

    def bgmm_clustering(self, k):
        bgmm = BayesianGaussianMixture(n_components=k, covariance_type=self.covariance_type,
                              init_params=self.init_params, max_iter=1000).fit(self.data)

        # data points assigned to a cluster
        labels = bgmm.predict(self.data)
        centroids = np.zeros((3, 3))
        return labels, centroids

    def find_clusters(self):
        optimal_k = self.find_optimal_k()
        print(f"Optimal k: {optimal_k}")
        labels, centroids = self.bgmm_clustering(optimal_k)
        return labels, centroids, self.data

class Dbscan:
    def __init__(self, data):
        self.data = np.transpose(data)

    def dbscan_clustering(self):
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(self.data)

        dbscan = DBSCAN(eps=0.5, min_samples=5)
        labels = dbscan.fit_predict(data_scaled)

        centroids = np.zeros((3, 3))
        return labels, centroids

    def find_clusters(self):
        labels, centroids = self.dbscan_clustering()
        return labels, centroids, self.data
