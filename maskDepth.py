import numpy as np
import PIL.Image as Image
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from kneed import KneeLocator
from mpl_toolkits.mplot3d import Axes3D
import random

def normalize_array(arr):
    min_val = np.min(arr)
    max_val = np.max(arr)
    if max_val==min_val:
        return np.zeros(arr.shape)
    normalized_arr = (arr - min_val) / (max_val - min_val) * 255
    return normalized_arr

# get masks from segmentations
def get_segmentation_masks(img):
    masks = []
    I = img.convert('L')
    I = np.asarray(I)
    for c in np.unique(I):
        segmentation = I == c
        # test = np.uint8(segmentation * 255)
        masks.append(segmentation)
    return masks

# mask depth image with segmentations
def get_masked_depth(depth_img, masks):
    depth_array = np.asarray(depth_img)
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
        masked_depth.save("aachen_000000_000019_disparity_mask_" + str(i) + ".jpg")

def create_point_clouds(masked_depths):
    point_clouds = []
    for mask in masked_depths:
        non_zero = np.nonzero(mask)
        point_cloud = np.array([non_zero[0], non_zero[1], mask[non_zero[0],non_zero[1]]])
        point_clouds.append(point_cloud)
    return point_clouds


def find_optimal_k(data, max_k):
    distortions = []
    for k in range(1, max_k+1):
        kmeans = KMeans(n_clusters=k, random_state=0, n_init=max_k).fit(data)
        distortions.append(kmeans.inertia_)

    # Plot the elbow curve
    plt.plot(range(1, max_k + 1), distortions, marker='o')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Distortion')
    plt.title('Elbow Curve')
    #plt.show()

    kn = KneeLocator(range(1,max_k+1),distortions, curve='convex', direction= 'decreasing')
    # Find the optimal k using the elbow point
    return kn.knee


def kmeans_clustering(data, k, current_num_instances ):
    kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
    instance_mask=np.zeros((320,320))#resolution is hard coded at (320,320)
    for index, point in enumerate(data):
        instance_mask[int(point[0]),int(point[1])]=kmeans.labels_[index]+current_num_instances

    colorMask = grayscale_to_random_color(instance_mask, k+current_num_instances).astype(np.uint8)
    #Image.fromarray(colorMask).convert('RGB').show()
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    return labels, centroids, instance_mask



def grayscale_to_random_color(grayscale,num_colors):
    color_list=[]
    for i in range(num_colors):
        color = list(np.random.choice(range(256), size=3))
        color_list.append(color)
    result = np.zeros((320,320,3))
    for i in  range(319):
        for j in range(319):
            if int(grayscale[i,j])==0:
                result[i, j] = (0,0,0)
            else:
                result[i,j]=color_list[int(grayscale[i,j])]
    return result



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

if __name__ == '__main__':
    img_path = "aachen_000000_000019_gtFine_color.png"
    depth_path = "aachen_000000_000019_disparity.png"

    masks = get_segmentation_masks(img_path)
    masked_depths = get_masked_depth(depth_path, masks)
    #save_masks(masked_depths)
    point_clouds = create_point_clouds(masked_depths)

    # K-Means Test
    data = np.transpose(point_clouds[3])
    max_k = 10  # Maximum value of k to consider
    optimal_k = find_optimal_k(data, max_k)
    print(f"Optimal value of k: {optimal_k}")
    labels, centroids = kmeans_clustering(data, optimal_k)
    print("Cluster labels:", labels)
    print("Centroids:", centroids)
    visualize_clusters(data, labels, centroids)


