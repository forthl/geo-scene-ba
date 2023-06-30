import numpy as np
import PIL.Image as Image
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from kneed import KneeLocator
from scipy.spatial.distance import cdist
from sklearn.mixture import GaussianMixture

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
    depth_array = (np.asarray(D) -1) / 256

    masked_depths = []
    for mask in masks:
        seg_masked = np.where(mask, depth_array, 0)
        masked_depth = np.uint8(seg_masked)
        masked_depths.append(masked_depth)
    return masked_depths


def save_masks(masked_depths):
    for i, d in enumerate(masked_depths):
        masked_depth = Image.fromarray(d).convert('RGB')
        masked_depth.save(
            "aachen_000000_000019_disparity_mask_" + str(i) + ".jpg")


def create_point_clouds(masked_depths):
    # parameters taken from aachen_000000_000019_camera.json
    fx = 2262.52
    fy = 2265.3017905988554
    cx = 1096.98
    cy = 513.137
    b = 0.209313

    point_clouds = []
    for mask in masked_depths:
        non_zero = np.nonzero(mask)
        depth = b * fx / (mask[non_zero[0], non_zero[1]])
        point_cloud = np.array(
            [non_zero[0], non_zero[1], mask[non_zero[0], non_zero[1]]])
        #point_cloud = np.array([(non_zero[0] - cx) * depth / fx, (non_zero[1] - cy) * depth / fy, b * fx / depth])
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
    ks = np.arange(1, max_k)
    bics = []
    for k in ks:
        gmm = GaussianMixture(n_components=k, init_params='kmeans')
        gmm.fit(data)
        bics.append(gmm.bic(data))

    # Plot the data
    fig, ax = plt.subplots()
    ax.plot(ks, bics)
    ax.set_xlabel(r'Number of clusters, $k$')
    ax.set_ylabel('BIC')
    ax.set_xticks(ks);

    diff = [x - bics[i - 1] for i, x in enumerate(bics)][1:]

    return diff.index(min(diff)) + 2

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

def find_clusters_kmeans(point_cloud, max_k, optimal_k_method):
    data = np.transpose(point_cloud)
    optimal_k = find_optimal_k(data, max_k, optimal_k_method)
    print(f"Optimal k: {optimal_k}")
    labels, centroids = kmeans_clustering(data, optimal_k)
    return labels, centroids, data


def distance(x, y, centroids):
    for i, centroid in enumerate(centroids):
        dist = np.linalg.norm(np.array([x, y]) - np.array([centroid[0], centroid[1]]))
    return 0

def visualize_clusters(masked_depths, sampleIdx):

    sample = masked_depths[sampleIdx] # 1 for cars, 10 for street lamps
    x = create_point_clouds([sample])
    x = x[0]

    labels, centroids, data = find_clusters_kmeans(x, 20, optimal_k_method='bic')
    c = np.transpose(centroids)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(x[0], x[1], x[2], c=labels);
    ax.scatter3D(c[0], c[1], c[2], 'black');
    plt.xlim(0, 1024)
    plt.ylim(0, 2048)
    ax.set_zlim(0, 255)
    plt.show()


if __name__ == '__main__':
    img_path = "tmp_data/aachen_000000_000019_gtFine_color.png"
    depth_path = "tmp_data/aachen_000000_000019_disparity.png"

    sampleIdx = 1 # 1 for cars, 10 for street lamps, 6 for street

    masks = get_segmentation_masks(img_path)
    Image.fromarray(masks[sampleIdx]).show()
    masked_depths = get_masked_depth(depth_path, masks)
    # save_masks(masked_depths)
    visualize_clusters(masked_depths, sampleIdx)
    point_clouds = create_point_clouds(masked_depths)


    x = False
    if x:
        mask = masks[3]
        point_cloud = point_clouds[3]
        labels, centroids, data = find_clusters(point_cloud, 20, optimal_k_method='bic')

        show_images = False
        if show_images:
            for j in range(len(centroids)):
                tmp = np.zeros(mask.shape)
                for i in range(len(data)):
                    if labels[i] == j:
                        tmp[int(data[i][0]), int(data[i][1])] = 255
                tmp = Image.fromarray(tmp).convert('RGB')
                tmp.show()

