import numpy as np
import PIL.Image as Image
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from kneed import KneeLocator
from scipy.spatial.distance import cdist
from sklearn.mixture import GaussianMixture

def normalize_array(arr):
    min_val = np.min(arr)
    max_val = np.max(arr)
    normalized_arr = (arr - min_val) / (max_val - min_val) * 255
    return normalized_arr

# get masks from segmentations
def get_segmentation_masks(img_path):
    masks = []
    I = Image.open(img_path).convert('L')
    I = np.asarray(I)
    for c in np.unique(I):
        segmentation = I == c
        # test = np.uint8(segmentation * 255)
        masks.append(segmentation)
    return masks

# mask depth image with segmentations
def get_masked_depth(depth_path, masks):
    D = Image.open(depth_path)
    depth_array = np.asarray(D)
    depth_array = normalize_array(depth_array)
    masked_depths = []
    for mask in masks:
        seg_masked = np.where(mask, depth_array, 0)
        masked_depth = np.uint8(seg_masked)
        masked_depth = normalize_array(masked_depth)
        masked_depths.append(masked_depth)
        # masked_depth = Image.fromarray(masked_depth)
        # masked_depth.show()
    return masked_depths


def save_masks(masked_depths):
    for i, d in enumerate(masked_depths):
        masked_depth = Image.fromarray(d).convert('RGB')
        masked_depth.save(
            "aachen_000000_000019_disparity_mask_" + str(i) + ".jpg")


def create_point_clouds(masked_depths):
    point_clouds = []
    for mask in masked_depths:
        non_zero = np.nonzero(mask)
        point_cloud = np.array(
            [non_zero[0], non_zero[1], mask[non_zero[0], non_zero[1]]])
        point_clouds.append(point_cloud)
    return point_clouds

def optimal_k_elbow(data, max_k):
    distortions = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=0, n_init='auto').fit(data)
        distortions.append(kmeans.inertia_)

    # Find the optimal k
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
    #TODO
    return 1

def optimal_k_vrc(data, max_k):
    #TODO
    return 3

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


def kmeans_clustering(data, k):
    kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto').fit(data)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    return labels, centroids

def visualize_clusters(data, labels, centroids):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot data points
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels)

    # Plot centroids
    ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2],
               c='red', marker='x', s=200)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('K-means Clustering')

    plt.show()

def find_clusters(point_cloud, max_k, optimal_k_method):
    data = np.transpose(point_cloud)
    optimal_k = find_optimal_k(data, max_k, optimal_k_method)
    print(f"Optimal k: {optimal_k}")
    labels, centroids = kmeans_clustering(data, optimal_k)
    return labels, centroids, data


def distance(x, y, centroids):
    for i, centroid in enumerate(centroids):
        dist = np.linalg.norm(np.array([x, y]) - np.array([centroid[0], centroid[1]]))
    return 0

if __name__ == '__main__':
    img_path = "tmp_data/aachen_000000_000019_gtFine_color.png"
    depth_path = "tmp_data/aachen_000000_000019_disparity.png"

    masks = get_segmentation_masks(img_path)
    masked_depths = get_masked_depth(depth_path, masks)
    # save_masks(masked_depths)
    point_clouds = create_point_clouds(masked_depths)


    mask = masks[1]
    point_cloud = point_clouds[1]
    labels, centroids, data = find_clusters(point_cloud, 20, optimal_k_method='bic')

    for j in range(len(centroids)):
        tmp = np.zeros(mask.shape)
        for i in range(len(data)):
            if labels[i] == j:
                tmp[int(data[i][0]), int(data[i][1])] = 255
        tmp = Image.fromarray(tmp).convert('RGB')
        #tmp.show()

