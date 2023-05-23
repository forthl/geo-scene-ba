import torch

import numpy as np

from torchvision import transforms
from torch.utils.data import Dataset

from nuscenes.nuscenes import NuScenes
from utils.quad_tree import Point, Node, Quad

from utils.representation import pointcloudOnImage, plotGraph
from utils.data_utils import generatePointCloudFromRangeImage, generateQuadTreeFromRangeImage, generateRangeImageFromPointCloud, generateRangeImageFromTree


class nuScenes(Dataset):
    def __init__(self, nusc: NuScenes):
        self.nusc = nusc
        samples = [self.getSamplesForScene(
            scene['first_sample_token'], scene['last_sample_token']) for scene in self.nusc.scene]
        self.samples = [s for sample in samples for s in sample]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        sample = self.samples[idx]
        cam_front_data = self.nusc.get(
            'sample_data', sample['data']['CAM_FRONT_RIGHT'])
        lidar_data = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])

        pointcloud, colors, image = self.nusc.explorer.map_pointcloud_to_image(
            lidar_data['token'], cam_front_data['token']
        )

        transform = transforms.Compose([transforms.PILToTensor()])

        image = transform(image).permute(0, 2, 1).T
        pointcloud = torch.tensor(pointcloud)
        pointcloud[2, :] = torch.tensor(colors)

        return pointcloud, image, self.geoPointSegs(pointcloud, image)

    def collate_fn(data):
        min = 2**32
        images = []
        depths = []
        batchsize = len(data)

        # To discuss: currently only cutting away, should this be randomized?

        for (d, i) in data:
            min = d.shape[1] if d.shape[1] < min else min

        indices = range(0, min)
        for (d, i) in data:
            images.append(i)
            depths.append(d[:, indices])

        imgs = torch.tensor(
            (batchsize, images[0].shape[0], images[0].shape[1]), dtype=float)
        torch.cat(images, out=imgs)

        dps = torch.tensor(
            (batchsize, depths[0].shape[0], depths[0].shape[1]), dtype=float)
        torch.cat(depths, out=dps)

        return (imgs, dps)

    def getSamplesForScene(self, first_sample_token, last_sample_token, samples=[]):
        if first_sample_token == last_sample_token:
            return samples
        sample = self.nusc.get('sample', first_sample_token)
        samples.append(sample)
        self.getSamplesForScene(sample['next'], last_sample_token, samples)
        return samples

    def geoPointSegs(self, pointcloud, image):
        pointcloud = torch.round(pointcloud.T).long()
        range_image = generateRangeImageFromPointCloud(pointcloud, image.T.shape[1:])
        pointcloudOnImage(generatePointCloudFromRangeImage(range_image), image)
        range_without_ground = removeGround(range_image)
        pointcloudOnImage(generatePointCloudFromRangeImage(range_without_ground), image)

        segments = labelRangeImage(range_without_ground)
        
        pointcloudOnImage(generatePointCloudFromRangeImage(segments), image)


def removeGround(range_image):
    tans = generateQuadTreeFromRangeImage(range_image.T)
    pdist = torch.nn.PairwiseDistance(p=2)

    for c in range(range_image.shape[0]):
        c_idx = range_image.shape[0] - c - 1
        col_indeces = range_image[c_idx].nonzero()

        for i in range(col_indeces.shape[0]):
            idx = col_indeces.shape[0] - i - 1
            
            if i == 0:
                tans.insert(Node(Point(col_indeces[idx].item(), c), 1.5))
                continue


            idx_A = col_indeces[idx+1].item()
            idx_B = col_indeces[idx].item()

            z_A = range_image[c_idx][idx_A]
            z_B = range_image[c_idx][idx_B]

            A = torch.tensor([idx_A, z_A])
            B = torch.tensor([idx_B, z_B])
            C = torch.tensor([idx_A, z_B])

            tan = torch.atan2(pdist(B, C), pdist(A, C)).item()

            tans.insert(Node(Point(int(idx_A), c_idx), tan))

    labels = torch.zeros(range_image.shape)
   
    for c in range(range_image.shape[0]):
        idx = range_image.shape[0] - c - 1
        col_indeces = range_image[idx].nonzero()

        if col_indeces.shape[0] <= 1:
            continue

        labelGround(idx, col_indeces[-1].item(), labels, tans)

    no_ground = torch.abs(labels - 1) * range_image

    return no_ground


def labelGround(y, x, labels, tans: Quad):
    q = []
    p = Point(x, y)
    n = tans.search(p)
    
    if n is None:
        n = Node(p, 1.5)
    
    q.append(n)

    while len(q) > 0:
        node: Node = q[0]

        labels[node.pos.y][node.pos.x] = 1
        neighbors = neighborhood(node.pos, tans)

        for n in neighbors:
            if n in q:
                continue
            if labels[n.pos.y][n.pos.x] == 1:
                continue
            if node.data == 1.5:
                q.append(n)
                continue
                
            if np.abs(node.data - n.data) < 1.5:
                q.append(n)

        q = q[1:]


def labelRangeImage(range_image):
    tree = generateQuadTreeFromRangeImage(range_image, True)
    labels = generateQuadTreeFromRangeImage(range_image)

    l = 1

    nodes = tree.gather()

    for n in nodes:
        if labels.search(n.pos) is None:
            labelSegments(n, tree, labels, l)
            l += 1

    image = generateRangeImageFromTree(labels)
    
    return image

def labelSegments(n: Node, tree: Quad, labels: Quad, label):
    q = []
    q.append(n)

    while len(q) > 0:
        n: Node = q[0]

        labels.insert(Node(n.pos, label))

        nodes = neighborhood(n.pos, tree)

        for nn in nodes:
            if labels.search(nn.pos) is not None:
                continue

            d1 = torch.tensor(max(n.data.item(), nn.data.item()))
            d2 = torch.tensor(min(n.data.item(), nn.data.item()))

            phi = angle_between(np.array([n.pos.x, n.pos.y, n.data]),
                                np.array([nn.pos.x, nn.pos.y, nn.data])) if n.data > nn.data else angle_between(np.array(
                                    [nn.pos.x, nn.pos.y, nn.data]), np.array([n.pos.x, n.pos.y, n.data]))

            beta = np.arctan2(d2 * np.sin(phi), d1 - d2 * np.cos(phi))
            if beta > 0.174533 and nn not in q:
                q.append(nn)

        q = q[1:]


def neighborhood(point, tans: Quad):
    radius = 10

    neighbors = []
    neighbors = tans.findInRadius(point, radius)

    return neighbors


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
