import matplotlib.pyplot
import numpy as np
from PIL.Image import Image
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from kneed import KneeLocator
from sklearn.mixture import GaussianMixture
import open3d as o3d



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



def optimal_k_elbow(data, max_k):
    distortions = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=0, n_init='auto').fit(data)
        distortions.append(kmeans.inertia_)

    kn = KneeLocator(range(1, max_k + 1), distortions, curve='convex', direction='decreasing')
    return kn.knee


def optimal_k_bic(data, max_k):
    bics = []
    for k in range(1, max_k):
        gmm = GaussianMixture(n_components=k, init_params='kmeans')
        gmm.fit(data)
        bics.append(gmm.bic(data))

    return np.argmin(bics)


def optimal_k_ml(data, max_k):
    # TODO
    return 1


def optimal_k_vrc(data, max_k):
    vrcs = []
    SSE_1 = KMeans(n_clusters=1, init='k-means++', random_state=0).fit(data).inertia_
    vrcs.append(SSE_1)
    for k in range(2, max_k):
        if k == data.shape[0]:
            break
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=0).fit(data)
        SSE_k = kmeans.inertia_
        vrc_i = ((SSE_1 - SSE_k) / (k - 1)) / (SSE_k / (data.shape[0] - k))
        vrcs.append(vrc_i)
    plot_vrc = [np.nan, np.nan]
    for val in vrcs:
        plot_vrc.append(val)
    matplotlib.pyplot.plot(plot_vrc)
    return np.argmin(vrcs) + 2


def find_optimal_k(data, max_k, optimal_k_method='e'):
    optimal_k = 0
    if optimal_k_method == 'e':
        optimal_k = optimal_k_elbow(data, max_k)
    elif optimal_k_method == 'bic':
        optimal_k = optimal_k_bic(data, max_k)
    elif optimal_k_method == 'ml':
        optimal_k = optimal_k_ml(data, max_k)
    elif optimal_k_method == 'vrc':
        optimal_k = optimal_k_vrc(data, max_k)

    return optimal_k

def kmeans_clustering_with_preprocessing(point_cloud, image_shape, optimal_k_finder, preprocessing,
                                         normalization_range, remove_outliers=False, visualize_clusters_bool=False,
                                         visualize_instance_mask_bool=False):
    labels, centroids, instance_mask = [], [], []
    optimal_k = 0
    max_k = 10  # Maximum value of k to consider

    if remove_outliers:
        point_cloud = remove_point_cloud_outliers(point_cloud)

    if preprocessing == "std":
        stds = StandardScaler()
        standardized_point_cloud = stds.fit_transform(point_cloud)
        optimal_k = find_optimal_k(standardized_point_cloud, min(max_k, standardized_point_cloud.shape[0]),
                                   optimal_k_finder)
        labels, centroids, instance_mask = kmeans_clustering_standardized(point_cloud,image_shape, standardized_point_cloud,
                                                                          optimal_k)
        point_cloud = standardized_point_cloud  # for visualizing the clusters

    elif preprocessing == "nrm":
        point_cloud = np.transpose(point_cloud)
        normalized_depth = normalize_array_to_range(point_cloud[2], normalization_range)
        normalized_point_cloud = np.transpose([point_cloud[0], point_cloud[1], normalized_depth])
        optimal_k = find_optimal_k(normalized_point_cloud, min(max_k, normalized_point_cloud.shape[0]),
                                   optimal_k_finder)
        labels, centroids, instance_mask = kmeans_clustering_normalized(normalized_point_cloud,image_shape, optimal_k)

    elif preprocessing == "nothing":
        optimal_k = find_optimal_k(point_cloud, min(max_k, point_cloud.shape[0]), optimal_k_finder)
        labels, centroids, instance_mask = kmeans_clustering(point_cloud,image_shape,
                                                             optimal_k)

    if visualize_instance_mask_bool:
        visualize_instance_mask(instance_mask, image_shape)

    if visualize_clusters_bool:
        visualize_clusters(point_cloud, labels, centroids)

    return labels, centroids, instance_mask


def kmeans_clustering(data, image_shape, k):
    kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
    instance_mask = np.zeros(image_shape)
    kmeans.labels_ = kmeans.labels_ + 1  # move labels from range (0,num_clusters-1) to range(1,num_clusters) for better visualization
    for index, point in enumerate(data):
        instance_mask[int(point[0]), int(point[1])] = kmeans.labels_[index]

    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    return labels, centroids, instance_mask


def kmeans_clustering_normalized(data,image_shape, k):
    kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
    instance_mask = np.zeros(image_shape)
    kmeans.labels_ = kmeans.labels_ + 1  # move labels from range (0,num_clusters-1) to range(1,num_clusters) for better visualization
    for index, point in enumerate(data):
        instance_mask[int(point[0]), int(point[1])] = kmeans.labels_[index]

    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    return labels, centroids, instance_mask


def kmeans_clustering_standardized(data, image_shape, standardized_data, k):
    kmeans = KMeans(n_clusters=k, random_state=0).fit(standardized_data)
    instance_mask = np.zeros(image_shape)
    kmeans.labels_ = kmeans.labels_ + 1  # move labels from range (0,num_clusters-1) to range(1,num_clusters) for better visualization
    for index, point in enumerate(data):
        instance_mask[int(point[0]), int(point[1])] = kmeans.labels_[index]

    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    return labels, centroids, instance_mask


def normalize_array_to_range(arr, range):
    min_val = np.min(arr)
    max_val = np.max(arr)
    if max_val == min_val:
        return np.zeros(arr.shape)
    normalized_arr = ((arr - min_val) / (max_val - min_val)) * range
    return normalized_arr


def visualize_instance_mask(instance_mask, image_shape):
    colorMask = grayscale_to_random_color(instance_mask, image_shape).astype(np.uint8)
    Image.fromarray(colorMask).convert('RGB').show()


def remove_point_cloud_outliers(point_cloud):
    # removes all points that don't have less than np_points in their neighborhood of radius
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    pcd = pcd.voxel_down_sample(voxel_size=0.01)

    # Radius outlier removal:
    pcd_rad, ind_rad = pcd.remove_radius_outlier(nb_points=50, radius=1)
    outlier_rad_pcd = pcd.select_by_index(ind_rad, invert=True)
    outlier_rad_pcd.paint_uniform_color([1., 0., 1.])
    pcdnp = np.asarray(pcd_rad.points)

    return pcdnp, ind_rad



def visualize_clusters(data, labels, centroids):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot data points
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels)

    # Plot centroids
    #ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2],
    #           c='red', marker='x', s=200)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('K-means Clustering')

    plt.show()



def grayscale_to_random_color(grayscale, image_shape=(320, 320)):
    color_list = []
    color_list.append((0, 0, 0))  # 0 values are not part of any instance thus black
    for i in range(1000):
        color = list(np.random.choice(range(256), size=3))
        color_list.append(color)
    result = np.zeros((image_shape[0], image_shape[1], 3))
    for i in range(image_shape[0]):
        for j in range(image_shape[1]):
            result[i, j] = color_list[int(grayscale[i, j])]
    return result
