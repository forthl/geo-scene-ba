import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import clusterAlgorithms
import cv2
from plyfile import PlyData, PlyElement


# get masks from segmentations
def get_segmentation_masks(img_path):
    masks = []
    I = Image.open(img_path).convert('L')
    I = np.asarray(I)
    for c in np.unique(I):
        segmentation = I == c
        masks.append(segmentation)
    return masks


# mask depth image with segmentations
def get_masked_disparity(disparity_path, masks):
    D = Image.open(disparity_path)
    disparity_array = np.asarray(D)

    masked_disparities = []
    for mask in masks:
        seg_masked = np.where(mask, disparity_array, 0)
        masked_disparity = np.uint8(seg_masked)
        masked_disparities.append(masked_disparity)
    return masked_disparities

# gets a depth map for each mask
def get_masked_depth(disparity_path, masks):
    disparity_array = cv2.imread(disparity_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    disparity_array[disparity_array > 0] = (disparity_array[disparity_array > 0] - 1.0) / 256.0

    masked_depths = []
    for mask in masks:
        seg_masked = np.where(mask, disparity_array, 0)
        masked_depths.append(seg_masked)
    return masked_depths


# saves masks on hard drive
def save_masks(masked_depths):
    for i, d in enumerate(masked_depths):
        masked_depth = Image.fromarray(d).convert('RGB')
        masked_depth.save(
            "aachen_000000_000019_disparity_mask_" + str(i) + ".jpg")


# creates point cloud out of masked depth images
def create_point_clouds(masked_depths):
    point_clouds = []
    for mask in masked_depths:
        non_zero = np.nonzero(mask)
        point_cloud = np.array(
            [non_zero[0], non_zero[1], mask[non_zero[0], non_zero[1]]])
        point_clouds.append(point_cloud)
    return point_clouds

# creates point cloud out of projected masked depth images
def create_projected_point_clouds(masked_depths):
    point_clouds = []
    for mask in masked_depths:
        point_cloud = project_disparity_to_3d(mask)
        point_clouds.append(point_cloud)
    return point_clouds


# visualizes clusters of data
def visualize_clusters(labels, centroids, data):
    c = np.transpose(centroids)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(data[0], data[1], data[2], c=labels);
    #ax.scatter3D(c[0], c[1], c[2], 'black');
    plt.xlim(0, 1024)
    plt.ylim(0, 2048)
    ax.set_zlim(0, 255)
    plt.show()


# computes cluster centroids and labels of each datapoint
def get_clusters(masked_depths, sampleIdx):
    sample = masked_depths[sampleIdx]  # 1 for cars, 10 for streetlamps
    data = create_point_clouds([sample])
    data = data[0]

    # cl = clusterAlgorithms.Kmeans(data=data, max_k=20)
    cl = clusterAlgorithms.GaussianMixtureModel(data=data, max_k=20)
    # cl = clusterAlgorithms.BayesianGaussianMixtureModel(data=data, max_k=20)
    # cl = clusterAlgorithms.Spectral(data=data, max_k=20)
    # cl = clusterAlgorithms.Dbscan(data=data)
    labels, centroids, _ = cl.find_clusters()
    return labels, centroids, data

def get_projected_clusters(masked_depths, sampleIdx):
    data = create_projected_point_clouds([masked_depths[sampleIdx]])
    data = np.transpose(data[0])

    # cl = clusterAlgorithms.Kmeans(data=data, max_k=20)
    # cl = clusterAlgorithms.GaussianMixtureModel(data=data, max_k=20)
    cl = clusterAlgorithms.BayesianGaussianMixtureModel(data=data, max_k=20)
    # cl = clusterAlgorithms.Spectral(data=data, max_k=20)
    # cl = clusterAlgorithms.Dbscan(data=data)
    labels, centroids, _ = cl.find_clusters()
    return labels, centroids, data

def project_disparity_to_3d(disparity_map):
    focal_length_x = 2262.52
    focal_length_y = 2265.3017905988554
    cx = 1096.98
    cy = 513.137
    baseline = 0.209313

    height, width = disparity_map.shape

    # Generate a grid of pixel coordinates
    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))

    # Compute the depth from disparity
    depth = (baseline * focal_length_x) / disparity_map

    # Filter out points with disparity value of 0
    valid_indices = np.where(disparity_map != 0)
    depth = depth[valid_indices]
    points_x = (grid_x[valid_indices] - cx) * depth / focal_length_x
    points_y = (grid_y[valid_indices] - cy) * depth / focal_length_y
    points_z = depth

    # Stack the coordinates into a point cloud
    point_cloud = np.stack((points_x, points_y, points_z), axis=-1)

    write_ply(point_cloud, 'pointcloud.ply', False)

    return point_cloud

# exports point cloud as .ply file
def write_ply(points, filename, text=True):
    """ input: Nx3, write points to filename as PLY format. """
    points = [(points[i, 0], points[i, 1], points[i, 2]) for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=text).write(filename)

def visualize_projected_clusters(labels, centroids, data):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    c = np.transpose(centroids)
    ax = plt.axes(projection='3d')
    ax.scatter3D(data[0], data[1], data[2], c=labels);
    #ax.scatter3D(c[0], c[1], c[2], 'black');
    plt.xlim(-6, 6)
    plt.ylim(-2, 2)
    ax.set_zlim(0, 32)

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Point Cloud')

    plt.show()

if __name__ == '__main__':
    img_path = "tmp_data/aachen_000000_000019_gtFine_color.png"
    disparity_path = "tmp_data/aachen_000000_000019_disparity.png"

    sampleIdx = 1  # 1 for cars, 10 for streetlamps, 6 for street
    masks = get_segmentation_masks(img_path)
    masked_disparities = get_masked_disparity(disparity_path, masks)
    masked_depths = get_masked_depth(disparity_path, masks)

    #labels1, centroids1, data1 = get_clusters(masked_disparities, sampleIdx)
    labels2, centroids2, data2 = get_projected_clusters(masked_depths, sampleIdx)

    #visualize_clusters(labels1, centroids1, data1)
    visualize_projected_clusters(labels2, centroids2, data2)