import torch

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
    idc = range_image.nonzero()
    
    pc = torch.tensor([[i[0], i[1], range_image[i[0]][i[1]]] for i in idc])
    
    return pc.T