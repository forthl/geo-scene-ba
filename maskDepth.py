import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import clusterAlgorithms


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
def get_masked_disparity(disparity_path, masks):
    D = Image.open(disparity_path)
    disparity_array = np.asarray(D)

    masked_disparities = []
    for mask in masks:
        seg_masked = np.where(mask, disparity_array, 0)
        masked_disparity = np.uint8(seg_masked)
        masked_disparities.append(masked_disparity)
    return masked_disparities


# saves masks on hard drive
def save_masks(masked_depths):
    for i, d in enumerate(masked_depths):
        masked_depth = Image.fromarray(d).convert('RGB')
        masked_depth.save(
            "aachen_000000_000019_disparity_mask_" + str(i) + ".jpg")


# creates point cloud out of masked depth images
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
    # cl = clusterAlgorithms.GaussianMixtureModel(data=data, max_k=20)
    # cl = clusterAlgorithms.Spectral(data=data, max_k=20)
    cl = clusterAlgorithms.Dbscan(data=data)
    labels, centroids, _ = cl.find_clusters()
    return labels, centroids, data


def visualize_data(data):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(data[0], data[1], data[2]);
    plt.xlim(-150, 350)
    plt.ylim(-150, 350)
    #ax.set_zlim(0, 255)
    plt.show()

def test(disparity):

    pc = []
    for i, row in enumerate(disparity):
        for j, disp in enumerate(row):
            if disp > 0.0:
                pc.append([i, j, disp])

    pct = np.transpose(pc)
    visualize_data(pct)

    ppct = np.transpose(project(disparity))
    visualize_data(ppct)

def project(disparity):
    fx = 2262.52
    fy = 2265.3017905988554
    cx = 1096.98
    cy = 513.137
    b = 0.209313


    pc = []
    for i, row in enumerate(disparity):
        for j, disp in enumerate(row):
            if disp > 0.0:
                pc.append([(i - cx) * (b * fx / disp) / fx, (j - cy) * (b * fx / disp) / fy, b * fx / disp])
    return pc

if __name__ == '__main__':
    img_path = "tmp_data/aachen_000000_000019_gtFine_color.png"
    disparity_path = "tmp_data/aachen_000000_000019_disparity.png"

    sampleIdx = 10  # 1 for cars, 10 for streetlamps, 6 for street
    masks = get_segmentation_masks(img_path)
    masked_disparities = get_masked_disparity(disparity_path, masks)


    Image.fromarray(masks[sampleIdx]).show()

    labels, centroids, data = get_clusters(masked_disparities, sampleIdx)
    visualize_clusters(labels, centroids, data)



    x = False
    if x:
        point_clouds = create_point_clouds(masked_depths)
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

