import matplotlib.pyplot
import numpy as np
import PIL.Image as Image
from sklearn.cluster import KMeans
import sklearn as skl
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from kneed import KneeLocator
from mpl_toolkits.mplot3d import Axes3D
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
import random
import clusterAlgorithms
import open3d as o3d
import cv2


def normalize_array_to_range(arr, range):
    min_val = np.min(arr)
    max_val = np.max(arr)
    if max_val == min_val:
        return np.zeros(arr.shape)
    normalized_arr = ((arr - min_val) / (max_val - min_val)) * range
    return normalized_arr

# get masks from segmentations from gep_seg
def get_segmentation_masks_geo_seg(geo_seg):
    masks = []
    for c in np.unique(geo_seg):
        segmentation = geo_seg == c
        masks.append(segmentation)
    return masks

# get masks from segmentations
def get_segmentation_masks(img):
    masks = []
    I = img.convert('L')
    I = np.asarray(I)
    for c in np.unique(I):
        segmentation = I == c
        segmentation = cv2.resize(segmentation.astype('uint8'), (2048, 1024), interpolation=cv2.INTER_NEAREST)
        masks.append(segmentation)
    return masks


# mask depth image with segmentations
def get_masked_depth(depth_img, masks):
    depth_array = np.asarray(depth_img).astype(np.float32)
    depth_array[depth_array > 0] = (depth_array[depth_array > 0] - 1.0) / 256.0
    depth_array = cv2.resize(depth_array, (2048, 1024), cv2.INTER_CUBIC)
    # depth_array = normalize_array_to_range(depth_array)
    masked_depths = []
    for mask in masks:
        seg_masked = np.where(mask, depth_array, 0)
        masked_depths.append(seg_masked)
    return masked_depths

def create_point_clouds(masked_depths):
    point_clouds = []
    for mask in masked_depths:
        non_zero = np.nonzero(mask)
        point_cloud = np.array([non_zero[0], non_zero[1], mask[non_zero[0], non_zero[1]]])
        point_cloud = np.transpose(point_cloud)
        point_clouds.append(point_cloud)
    return point_clouds

def create_projected_point_clouds(masked_depths):
    point_clouds = []
    for mask in masked_depths:
        point_cloud = project_disparity_to_3d(mask)
        point_clouds.append(point_cloud)
    return point_clouds

def unproject_point_cloud(data):
    focal_length_x = 2262.52
    cx = 1096.98
    cy = 513.137
    focal_length_y = 2265.3017905988554

    data = np.transpose(data)

    for point in data:
        point[0] = int(round((point[0] * focal_length_x / point[2]) + cx))
        point[1] = int(round((point[1] * focal_length_y / point[2]) + cy))

    return data.astype('int')

def project_disparity_to_3d(disparity_map):
    focal_length_x = 2262.52
    focal_length_y = 2265.3017905988554
    cx = 1096.98
    cy = 513.137
    baseline = 0.209313

    height, width = disparity_map.shape

    # Generate a grid of pixel coordinates
    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))

    # Filter out points with disparity value of 0
    valid_indices = np.where(disparity_map != 0)
    depth = (baseline * focal_length_x) / disparity_map[valid_indices]
    points_x = (grid_x[valid_indices] - cx) * depth / focal_length_x
    points_y = (grid_y[valid_indices] - cy) * depth / focal_length_y
    points_z = depth

    # Stack the coordinates into a point cloud
    point_cloud = np.stack((points_x, points_y, points_z), axis=-1)

    return point_cloud


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


def DBSCAN_clustering(data, image_shape, epsilon, min_samples):
    dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
    instance_mask = np.zeros(image_shape)
    dbscan = dbscan.fit(data)
    labels = dbscan.labels_
    labels += 1  # this converts the label range from(-1,num_clusters-1) to (0,num_clusters) in order for each instance to have unique id in the merged instance mask
    # TODO -1 represents noise but for right now we treat it as missing data
    for index, point in enumerate(data):
        instance_mask[int(point[0]), int(point[1])] = labels[index]

    #colorMask = grayscale_to_random_color(instance_mask, num_clusters).astype(np.uint8)
    #Image.fromarray(colorMask).convert('RGB').show()

    return labels, instance_mask

def BGMM_Clustering(data, image_shape, depth_image, max_k=20):

    # remove lonely points to denoise point cloud
    filtered_data, inliersIdx = remove_point_cloud_outliers(data)

    if len(filtered_data < 2):
        data = np.transpose(data)
    else:
        data = np.transpose(filtered_data)

    # cluster points
    cl = clusterAlgorithms.BayesianGaussianMixtureModel(data=data, max_k=20)
    labels, _, _ = cl.find_clusters()
    labels += 1

    # unproject points to find original image coordinates
    data = unproject_point_cloud(data)

    # assign labels to instance_mask
    instance_mask = np.zeros((1024, 2048))
    for i, point in enumerate(data):
        instance_mask[point[1], point[0]] = (labels[i] + 1) * 255 / len(np.unique(labels))
    instance_mask_small = cv2.resize(instance_mask, (320, 320), cv2.INTER_NEAREST)
    return labels, instance_mask_small

def remove_point_cloud_outliers(point_cloud):
    # removes all points that don't have less than np_points in their neighborhood of radius
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    pcd = pcd.voxel_down_sample(voxel_size=0.01)

    # Radius outlier removal:
    pcd_rad, ind_rad = pcd.remove_radius_outlier(nb_points=15, radius=1)
    outlier_rad_pcd = pcd.select_by_index(ind_rad, invert=True)
    outlier_rad_pcd.paint_uniform_color([1., 0., 1.])
    pcdnp = np.asarray(pcd_rad.points)

    return pcdnp, ind_rad

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


def visualize_instance_mask(instance_mask, image_shape):
    colorMask = grayscale_to_random_color(instance_mask, image_shape).astype(np.uint8)
    Image.fromarray(colorMask).convert('RGB').show()

