import torch

import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt

from src.utils.quad_tree import Point, Node, Quad


def generateRangeImageFromPointCloud(pointcloud, shape):
    range_image = torch.zeros(shape)

    for p in pointcloud:
        range_image[p[0]][p[1]] = p[2].float()

    return range_image
    
def generateRangeImageFromTree(tree: Quad):
    nodes = tree.gather()
    
    range_image = torch.zeros((tree.bot_right.x, tree.bot_right.y))
    
    for n in nodes:
        if n.pos is not None:
            range_image[n.pos.x][n.pos.y] = n.data 

    return range_image


def generateQuadTreeFromRangeImage(range_image, insert=False):
    tree = Quad(Point(), Point(
        range_image.shape[0] + 1, range_image.shape[1] + 1))

    if insert:
        for y in range(range_image.shape[0]):
            rows = range_image[y].nonzero()
            for x in rows:
                tree.insert(Node(Point(y, x.item()), range_image[y][x.item()]))

    return tree

def generatePointCloudFromRangeImage(range_image):
    depth_instensity = np.array(256 * range_image / 0x0fff,
                            dtype=np.uint8)
    
    iio.imwrite('outputs/grayscale.png', depth_instensity)
    
    # fig, axs = plt.subplots(1, 2)
    # axs[0].imshow(range_image, cmap="gray")
    # axs[0].set_title('Depth image')
    # axs[1].imshow(depth_instensity, cmap="gray")
    # axs[1].set_title('Depth grayscale image')
    # plt.show()
    
    pcd = []
    
    # height, width = range_image.shape
    height, width = 512, 1024
    for i in range(height):
       for j in range(width):
           z = depth_instensity[i][j]
           x = j
           y = i
           if z > 0:
            pcd.append([x, y, z])
    
    pc = torch.tensor(pcd).float()
    
    return pc.T

def imageFromPointsAndClassification(pc, classification, range_image):    
    img = np.zeros((513, 1025))
    
    print(pc.shape)
    
    pc = pc.int().cpu().numpy()
    
    for i in range(pc.shape[2]):
        img[pc[0, 1, i].data, pc[0, 0, i].data] = classification[i]
    
    fig, axs = plt.subplots(1, 2)
    
    axs[0].imshow(img)
    axs[0].set_title('Instances')
    axs[1].imshow(range_image[:513, :1025])
    axs[1].set_title('Depth')
    plt.show()